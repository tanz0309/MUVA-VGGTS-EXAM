# VGGT-S：官方训练函数与 MUVA 数据接口

更新于2026-11-06。当前不加入分块优先、图聚类、自适应提示等创新模块。统一入口 `run.sh ... vggts` 已切换到本页版本，旧适配入口已停用。

## 官方来源及实际调用

- [作者仓库](https://github.com/buaa-colalab/VGGT-S)，固定提交 `7ba82679d662d42171b95f1977f01328f1b18630`。
- 本地源码：`/root/datasets/MUVA_experiments/vendor/VGGT-S-official`。
- [VGGT 骨干](https://github.com/facebookresearch/vggt)，固定提交 `a288dd0f14786c93483e45524328726ab7b1b4ce`，本地 `vendor/vggt-official`。
- VGGT-S 的模型和 `src/main_exp/hybrid.py` 不改动，训练直接调用作者的 `train_epoch`。
- VGGT 仅按 VGGT-S README 要求给 track head 增加 `feat=True` 特征返回接口。
- 入口：`scripts/run_vggts_official.py`；数据接口：`scripts/vggts_official_data.py`；有效配置：`configs/vggts_official.json`。

**真实跨视角配对监督、50→40→5 提示、掩码细化、细化分支只对末轮反传、20×Focal+Dice，都是作者已有机制。** 不能将它们说成本地创新，也不能将单图形变自监督误说成论文唯一训练方式。

## 预处理与预测点

1. 从参考真实可见掩码采样50点。使用作者 `pad_frame`：短边对齐14、右/底补 RGB=114、AREA 插值，构造518画布。
2. 冻结 VGGT 将点匹配到目标图。该阶段不读取目标 GT。
3. 把点还原到作者外部相机960×540坐标；MUVA原图为1920×1080，按实际宽高比例换算，首阶段短边为294。
4. 原生 `remove_outliers_median` 删除离中位数最远的10点，保留40点。
5. 原生 `cal_crop_size_3x` 确定裁剪尺寸，再调用目标裁剪函数。该函数不是简单乘3：边长小于50返回180，大于200返回540，中间为 `int(2.4*(extent-50)+180)`。
6. 原生 `cluster_points` 将40点聚为5点，保留作者内部 `random_state=0, n_init='auto'`，不再改为全局seed42或额外去重。
7. 参考侧调用外部参考图的 `clip_exo_mean` 和 `crop_and_resize_by_mask_safe(R=2)`。输入 Predictor，按原方法细化。

**默认按需生成预测点，无需提前预处理。** 训练读取一个图对时，先检查该图对的点缓存；缺失则用当前已加载、冻结的 VGGT 在 GPU 上生成50个预测点，写入缓存后立即进行裁剪和该批训练。训练期间的验证、独立验证和测试也采用同一规则，不扫描或等待整个划分的点全部生成。

缓存按协议、基础权重及参考RGB/参考掩码/目标RGB内容生成键，仍保存在 `cache/vggts_official_points/`。已有有效缓存直接复用，中断后重启和后续epoch无需重复生成；损坏或协议不符的缓存明确报错。独立 `prepare-points` 与 `check-data` 入口保留为可选工具。

在线生成复用同一个冻结骨干，使用 `no_grad`，不加载第二份 VGGT。DataLoader 保持 `num_workers=0`，在主进程顺序执行生成点和训练；生成点前后恢复Python、NumPy、PyTorch及当前CUDA随机状态，避免缓存命中与否改变训练随机流。全局seed仍为42，作者内部固定的KMeans参数不变。

终端首次生成及每累计生成25个图对打印 `POINTS_GENERATED`。结束或异常退出时，`point_cache_usage.json`记录本次按需新生成数与缓存复用数（包含预热和训练期间验证，不等于唯一案例数）；逐例`prompts/*.json`中的`first_stage_point_source`标记`generated_online`或`cache`。

## 训练

| 参数 | 当前设置 |
|---|---|
| 初始化 | 公开 VGGT 冻结；Predictor 随机初始化 |
| 数据 | train6224对；val2215对；test不参与选模 |
| batch | 单张4090，batch8；与作者每卡batch8一致，仍为单卡训练 |
| 轮数 | 12 |
| 优化 | AdamW，lr=5e-5，weight_decay=1e-4，梯度裁剪1 |
| 调度 | 第8、11轮衰减 |
| 损失 | 作者的20×Focal+Dice |
| 分支 | 同batch一半直接预测、一半细化；直接分支正常反传，细化分支仅末轮反传 |
| 最佳权重 | 每轮共同val的Visible mIoU |

保留作者有效训练案例规则：目标裁剪非空、参考前景面积达到阈值，失败最多重试10次。替换记录写入 `training_retries.jsonl`。这只影响训练采样，验证/测试不按 GT 过滤失败案例。

`training.jsonl` 记录每步总损失和直接/细化样本数，终端第1步及每25步打印；打印的是当前batch，不是25步平均。作者损失细项留在 `native_training.log`。

## 必须披露的适配

MUVA没有ego/exo相机标签，本接口组合作者外部参考图与外部目标图的处理函数，并恢复到原图坐标评价；不是复现原Ego-Exo4D榜单。单卡batch8仍不等价于作者的多卡总batch。

官方 PromptEncoder 原地修改来源点特征，可能触发 autograd 视图错误或重复累加位置编码。适配层在每次 Predictor 调用前克隆该输入；模型参数及损失不变，但不能声称与未修复代码逐位相同。单卡包装仅提供官方保存逻辑需要的 `.module` 接口。

新输出在 `outputs/vggts_official/`。旧 `outputs/vggts/` 的日志、权重保留，不作为当前基线，不接受旧协议或smoke权重用于正式恢复/评价。

## 命令与计时

完整命令见 [文档07](07_终端命令与结果查看.md)，当前检查证据见 [文档06](06_GPU可运行性检查报告.md)。直接执行 `GPU_ID=0 bash /root/datasets/MUVA_experiments/run.sh train vggts` 即可；训练后加载best.pth评价。首次处理未缓存图对会比后续轮次慢。

VGGT-S当前评价计时包含按需处理：缓存缺失时包含首阶段生成及缓存读写，命中时跳过首阶段计算。预热也可能生成首例缓存，因此整体是**受缓存命中情况影响的耗时**，不能当作统一的无缓存端到端测速。`summary.json/csv`的`timing`字段明确记录此范围，六项精度指标仍使用共同评估器。改为按需执行不改变模型配置、点缓存配方或正式权重格式。
