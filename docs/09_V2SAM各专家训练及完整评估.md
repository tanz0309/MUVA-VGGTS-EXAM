# V²-SAM：官方专家与 PCCS 补齐

更新于2026-11-17。四种入口复用 `/personal_workspace/TanZhuzhu/.conda/envs/v2sam`，所有网络计算在CUDA执行。

## 来源、版本与边界

[作者公开仓库](https://github.com/jaychempan/V2-SAM)固定于 `50e7c0584e852e1e16703c07e33c49e1d1bde635`，本地位于 `/root/datasets/MUVA_experiments/vendor/V2-SAM-official`，tracked源码未修改。作者README要求把Visual或Fusion目录命名为 `projects/v2sam`；本地用包别名兼容硬编码导入，同时明确指定Visual自身的SAM2扩展类，避免混用两个分支。

Visual/Fusion使用各自官方 `V2SAM.forward` 计算损失、官方 `V2SAM.predict` 生成预测。PCCS位于适配层 `scripts/pccs.py`，是依据论文3.5节和式(8)补齐的实现，**不是作者已公开的精确选择器代码**。网上检索依据和未明确细节见文档10。

| 入口 | 训练 | 权重 |
|---|---|---|
| `v2sam-anchor` | 免训练 | 公开SAM2解码器、DINOv3 |
| `v2sam` | Fusion专家 | 新训练VPMatcher、视觉投影、SAM2解码器 |
| `v2sam-visual` | Visual专家 | 独立训练另一套任务参数 |
| `v2sam-multi` | 选择器无额外训练 | 分别加载Fusion和Visual任务权重，保留原始Anchor解码器 |

只跑 `v2sam` 是 Fusion 单专家，不能写成完整 Multi-Experts。补齐PCCS会影响多专家结果的产生方式，但不改变两个专家的训练损失。

## 初始化与输入

SAM2-Hiera-L基础权重位于 `weights/sam2/sam2_hiera_large.pt`，SHA256为 `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`；DINOv3 ViT-L复用 `/root/INSID3/pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。

VPMatcher和视觉投影随机初始化；SAM2解码器从公开基础权重初始化，随后参与训练。编码器、DINO以及作者冻结的IoU/存在性头保持冻结。新训练不加载作者跨视角任务checkpoint，也不从smoke权重继续。

SAM2图像使用官方 `DirectResize(1024)`；提示掩码和训练监督按作者数据接口最近邻缩放到**1024×1024**。Anchor在768尺度提取DINO特征，每个对象最多1点，前景阈值0.6，空间分层阈值150，离群点比例0.25。

官方预测函数读取名为 `masks` 的字段获取输出尺寸。适配层提供全零1024×1024占位张量，不传目标GT。保留官方1024尺度阈值化、填洞，再最近邻恢复到目标原图。没有添加保留最大连通块或分块合并规则。

## 训练设置及4090适配

| 项目 | 当前设置 | 来源／差异 |
|---|---|---|
| Fusion轮数 | 24 | 官方Fusion配置 |
| Visual轮数 | 12 | 官方Visual配置；论文补充表统一写24，主实验按公开分支配置，需披露 |
| 单卡batch | 16 | 与作者每卡前向batch一致；新服务器4090配置 |
| 梯度累积 | 1 | 当前单卡每次直接更新；原作者为4，需披露差异 |
| 每次更新图对数 | 16 | 16×1；6224对恰好分为389次完整更新，不丢弃或复制样本 |
| 优化器 | AdamW，lr4e-5，betas(0.9,0.999)，wd0.05 | 官方配置 |
| 混合精度 | BF16，动态loss scaler | 官方MMEngine AmpOptimWrapper |
| 裁剪 | 梯度范数1 | 官方配置 |
| 调度 | 5%迭代线性warmup，start_factor1e-5，之后cosine到0 | 官方MMEngine调度器 |
| 验证 | 两专家各自每2轮运行共同val，按Visible mIoU选best | Fusion与公开配置一致；Visual公开配置每1轮，当前MUVA入口改为每2轮，属本地频率差异 |

损失全部来自官方forward：视觉对比、结构约束mask与最终mask。训练驱动只替换MUVA迭代、统一评价和checkpoint IO，不是原 `tools/dist.sh` 分布式启动原样运行；优化、累积和调度由官方MMEngine类执行。

前向对比负样本池为batch16，梯度累积1次；有效更新batch16仍不同于作者16×4=64。学习率保持作者设置，不根据显存做未经验证的线性放大。按seed42正常打乱，不添加不同场景优先采样策略。记录每个 `microstep` 和实际optimizer `step`，每25个microstep打印。当前smoke-train仅执行1个microstep、16个图对、1次真实更新。

保存任务权重、优化器、AMP scaler、调度器、原生对比学习计数器和随机状态，支持完整epoch恢复。调度器按microstep更新；当前累积为1，microstep与实际optimizer更新同步记录。

## 训练中验证与完整Multi验证

Fusion与Visual都已随训练自动验证：当前MUVA入口每2轮用该专家本身评估完整val，结果在各自训练目录的`val_epoch_XX`，据此选择各自的`best.pth`。不需要手工命令才能产生这些单专家验证结果。

最终`evaluate v2sam-multi`评估的是Anchor、训练后的Fusion、训练后的Visual及论文式PCCS选择链路，需要同时加载两个专家权重；单专家的验证分数不能当作完整组合的分数。当前两个训练任务独立运行，训练入口尚未串接“双方结束后自动运行Multi评估”。单独执行Multi命令是本地任务组织方式，可以在不修改官方模型结构的情况下自动化，不是作者规定必须手工验证。

固定版本作者README要求分别训练Visual/Fusion，Anchor免训练；两个公开配置都包含训练中ValLoop，其中Fusion的`val_interval=2`、Visual为1。当前MUVA两者每2轮验证，必须如实记录Visual的频率差异，不能统称为完全相同的官方设置。当前运行不因此改配置或重启。

核对来源：[固定版本README](https://github.com/jaychempan/V2-SAM/blob/50e7c0584e852e1e16703c07e33c49e1d1bde635/README.md)、[Fusion配置](https://github.com/jaychempan/V2-SAM/blob/50e7c0584e852e1e16703c07e33c49e1d1bde635/projects/v2sam_fusion/configs/v2sam.py)、[Visual配置](https://github.com/jaychempan/V2-SAM/blob/50e7c0584e852e1e16703c07e33c49e1d1bde635/projects/v2sam_visual/configs/v2sam.py)。

## PCCS的具体定义

1. 分别运行Anchor、Visual、Fusion，得到三个目标预测掩码。
2. 将每个非空预测作为逆向Anchor的输入掩码，从目标图匹配回参考图。
3. 参考点使用前向Anchor在参考掩码内的采样点。对每个返回点，计算到参考点集合的最近欧氏距离，再取平均，除以参考图对角线。同一案例的归一化不改变排序。
4. 选择有效距离最小的专家，原样输出其掩码，不混合三个结果。

默认每对象1点，因此有效评分是两个点之间的距离。论文没有明确采样对应、点数不等、并列、无点失败时的全部细节；本地固定：点数不等采用平均最近点距离；空预测或无有限、图内返回点不评分；并列按Anchor→Visual→Fusion；全部无效固定回退Fusion。这些政策没有按目标GT优化。

逐案例 `prompts/*.json` 保存三个循环距离、参考点、返回点、选择结果、无效原因和回退标记。目标GT不参与选择，只在预测返回后计分。Visual/Fusion使用独立原生模型，按顺序前向；Anchor复用Fusion分支的冻结编码器和DINO，但切回单独保存的原始SAM2解码器。

## 检查与使用

新服务器4090前向、训练更新、保存重载和多专家GT隔离检查均通过，详情见文档06；正式训练、完整val/test均由用户启动，命令见文档07。新协议拒绝旧配置权重和smoke权重用于正式评价。

Anchor距离筛选的 `threshold+1` 在BF16中可能与threshold舍入成同值而导致循环不结束；仅此筛选函数强制GPU FP32，保留算法、阈值与采样方式，其余网络保持混合精度。该兼容修复和MUVA输入适配必须披露。
