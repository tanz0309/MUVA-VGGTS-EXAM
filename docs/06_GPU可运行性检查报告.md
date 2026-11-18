# 新服务器4090与旧服务器3090运行检查

## 新服务器验收（2026-11-17）

状态：MUVA数据、四个独立Conda环境和四种模型的数据接入均已通过检查。实际GPU为RTX4090，驱动580.178.04，驱动报告49,140 MiB，PyTorch可用48,485 MiB；网络计算均在CUDA上执行。四个环境的pip依赖检查均通过。

| 方法 | 本机检查范围 | 峰值allocated显存GiB | 结果 |
|---|---|---:|---|
| INSID3 | 2个真实案例前向 | 1.74 | 通过 |
| O-MaMa | 2个真实案例前向 | 3.45 | 通过 |
| O-MaMa | batch24，1次更新及保存重载 | 5.73 | 通过 |
| VGGT-S | 8个真实案例前向，包含按需生成点 | 5.63 | 通过 |
| VGGT-S | batch8，原生train_epoch更新及保存重载 | 16.61 | 通过 |
| V²-SAM Fusion | batch16×累积1，1次更新及保存重载 | 9.37 | 通过 |
| V²-SAM Visual | batch16×累积1，1次更新及保存重载 | 8.24 | 通过 |
| V²-SAM Multi | 2个真实案例、三个专家及PCCS | 3.89 | 通过 |

上述训练检查各只执行1次优化器更新，均核对参数确实更新、冻结骨干无梯度、权重可保存并严格重载。它们只用于验收接线和当前batch可运行性，不作为训练收敛、完整验证成绩或全数据最坏显存上限。正式训练、完整val/test由用户启动。

V²-SAM另外复用了已有GPU集成检查：Fusion和Visual逐参数加载各自检查权重；Anchor解码器逐参数等于公开基础权重；目标掩码占位值从全0改为全1时两专家输出不变；最终多专家输出为1080×1920，选择器不读取目标GT。集成检查峰值约3.91 GiB。

MUVA固定版本恢复到 `/root/datasets/MUVA`：四个归档SHA256通过，26,406张RGB与16,907张二值掩码全部解码通过，并重新用原标注RLE核对掩码。train/val/test为6224/2215/2008，场景和图片交集均为0，清单字节与迁移前一致，输入缺失为0。

六项精度主指标为Visible mIoU、Dice、Boundary F、CR@50、SCR@50、全分块覆盖成功率；复用同一个评价器，其现有6项边界检查通过。耗时、显存、Precision/Recall另作辅助信息。

四个主模型固定官方源码没有新增修改。O-MaMa仅在数据适配层补齐batch24，冻结描述符按图对提取后组成官方批量匹配；VGGT-S仅在数据迭代层统一首轮打乱种子为42。所有配置的公共seed为42，作者内部固定KMeans参数按文档03单独记录。

本机额外在O-MaMa、VGGT-S和V²-SAM各自Conda环境安装libgl/libglib，补齐OpenCV运行库；V²-SAM的MMCV2.1.0按CUDA12.4、架构8.9编译，MMEngine使用迁移包本地源码。完整环境包版本与检查证据统一保存到 [恢复记录](/root/to_migrate/_migration/reports/new_server_restore.json)，逐图检查见 [数据核验记录](/root/to_migrate/_migration/reports/muva_data_restore_verification.json)。

本次临时检查脚本、输出、检查权重和测试日志已经清理；保留正式数据、工程脚本、统一恢复记录和迁移前历史结果。正式命令见文档07。

## 以下为旧服务器3090历史检查

检查日期2026-11-06（北京时间）。模型均在各自Conda环境的RTX3090/CUDA上执行，没有CPU神经网络回退。此页列当前协议的小规模成功检查，正式实验另行记录。

## 实际覆盖

| 方法 | 模式 | 检查范围 | 峰值allocated显存GiB |
|---|---|---|---|
| INSID3 | smoke | 2个真实案例 | 1.61 |
| O-MaMa | smoke | 2个真实案例 | 3.45 |
| O-MaMa | smoke-train | 1次更新＋保存重载 | 2.59 |
| VGGT-S | smoke | 空缓存，在线生成点并推理2个真实案例 | 5.63 |
| VGGT-S | smoke-train | 在线生成2对预测点，1次更新＋保存重载 | 7.95 |
| V²-SAM Fusion | smoke-train | 4个microstep→1次更新＋保存重载 | 3.11 |
| V²-SAM Visual | smoke-train | 4个microstep→1次更新＋保存重载 | 1.87 |
| V²-SAM Multi+PCCS | smoke | 2个真实案例 | 3.89 |

V²-SAM Multi的两个案例实际执行Anchor、Visual、Fusion三个分支及三次逆向匹配，两个原生predict入口都被调用。Visual/Fusion任务头尚未正式训练；smoke指标不是性能结论。PCCS默认1点，在首例出现Anchor/Fusion同分时按固定优先级选择Anchor，审计中保留完整分数。

VGGT-S直接调用作者train_epoch，从未预计算的训练图对开始，实际在线生成2对预测点并完成1次更新；检查观测到直接预测有梯度、细化首轮无梯度、细化末轮有梯度，随机任务头确实更新并重载。推理检查从独立空缓存开始，新生成2对点，预热后的首例复用缓存1次。V²-SAM两个专家均核对了4次累积仅1次实际optimizer更新、非零对比损失、冻结参数无梯度、AMP/调度器状态及任务权重重载。O-MaMa按取消额外裁剪后的当前更新方式通过。

上表是PyTorch max_memory_allocated，不是nvidia-smi总占用或reserved显存；不据此承诺全数据集最坏显存上限。当前GPU有其他任务，smoke时间不用于论文速度比较。

## 成功输出

- INSID3 / smoke：[运行目录](../outputs/insid3/smoke/val_training_free_20261106T075529114429Z)。
- O-MaMa / smoke：[运行目录](../outputs/omama/smoke/val_new_task_head_20261106T075616750449Z)。
- O-MaMa / smoke-train：[运行目录](../outputs/omama/smoke-train/train_new_task_head_20261106T075636605876Z)。
- VGGT-S / smoke：[运行目录](../outputs/vggts_official/smoke/20261106T090006823310Z)。
- VGGT-S / smoke-train：[运行目录](../outputs/vggts_official/smoke-train/20261106T090539308203Z)。
- V²-SAM Fusion / smoke-train：[运行目录](../outputs/v2sam/smoke-train/train_new_task_head_20261106T074843719288Z)。
- V²-SAM Visual / smoke-train：[运行目录](../outputs/v2sam-visual/smoke-train/train_new_task_head_20261106T074946734370Z)。
- V²-SAM Multi+PCCS / smoke：[运行目录](../outputs/v2sam-multi/smoke/val_new_task_head_20261106T074611247525Z)。

## 数据、选择器与入口检查

- 数据仍为train6224/val2215/test2008，场景/图像跨划分交集0；不使用叠图输入网络。
- 新增PCCS的6项边界测试通过：最近返回点选择、不同点数与尺度、空候选、非有限/越界返回点、无参考点与并列策略、缺失专家报错。该测试只计算数组距离，不运行CPU网络。
- VGGT-S当前数据接口11项测试已通过，包含补边/裁剪/坐标、GT隔离、缓存身份、缺失点生成后复用、损坏缓存拒绝、随机状态保护、禁止DataLoader子进程生成GPU点及旧权重拒绝。另在真实CUDA设备检查随机状态保护，并核对新报告JSON/CSV的计时说明一致，证据见 [按需生成检查](../logs/vggts_on_demand_verification.json)。共同指标实现未改动，既有6项边界测试和Boundary F独立对照仍适用。
- 所有正式评价入口要求正确任务权重；免训练方法拒绝训练；新配置不接受旧配置checkpoint；smoke权重不冒充正式结果。

## 多专家接线与监督隔离实测

额外GPU集成检查通过：[v2sam_gpu_integration_check.json](../logs/v2sam_gpu_integration_check.json)。逐参数核对Fusion、Visual分别加载各自smoke训练权重；Anchor解码器与公开SAM2基础权重逐参数相同。把原生predict中的目标掩码占位值从全零改为全一并恢复相同随机状态，两个专家的二值输出均完全一致；不读取目标GT。多专家最终输出为1080×1920，峰值allocated约3.91GiB。

该检查只加载smoke权重验证接线，不经过正式评价入口，不用于报告性能。

## 修复、失败记录与限制

O-MaMa首次切换干净作者仓库时发现FastSAM不在该仓库内，首次检查因缺依赖路径失败；按作者README补齐独立官方FastSAM仓库并固定版本后，上表两项重查通过。失败目录保留，未计为成功。

VGGT-S在卡上可用显存不足时的一次早期合成检查OOM保留为失败；用户停止旧任务后，真实数据检查通过。V²-SAM距离分层筛选强制GPU FP32，其他网络保持BF16；PCCS缺少可获取的作者完整源码，采用明确标注的论文式实现，详见文档10。

本轮代理只执行小规模检查，未启动完整训练、超参数搜索或收敛验证。用户已自行完成INSID3的2215例val运行，记录在 [正式验证目录](../outputs/insid3/evaluate/val_training_free_20261106T082528466306Z)，不作为此页的smoke成绩。旧协议训练日志/权重原位保留，不能混入当前基线。

机器可读摘要：[configuration_verification.json](../logs/configuration_verification.json)。命令见文档07。
