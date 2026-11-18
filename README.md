# MUVA 四模型对比实验

INSID3、O-MaMa、VGGT-S、V²-SAM 在 MUVA 分块目标子集上的统一实验代码备份。保留现用适配代码、官方源码版本、环境规格及复现说明，用于版本管理和服务器恢复。

2026-11-18 从 RTX 4090 服务器首次整理。官方网络结构保持原方案；本地适配、兼容处理及 PCCS 补齐的边界见 [复现说明](docs/10_论文核对与复现边界.md)。仓库不包含数据集、模型权重、实验输出、环境实体或登录凭据。

## 模型与当前配置

| 方法 | 入口名称 | Conda 环境 | batch | epochs |
|---|---|---|---:|---:|
| INSID3 | `insid3` | `insid3-muva` | 1 | 免训练 |
| O-MaMa | `omama` | `omama` | 24 | 3 |
| VGGT-S | `vggts` | `vggts` | 8 | 12 |
| V²-SAM Fusion | `v2sam` | `v2sam` | 16 | 24 |
| V²-SAM Visual | `v2sam-visual` | 同上 | 16 | 12 |

公共 seed **42**；作者算法内部固定的 KMeans seed 0 保留并披露。六项主指标为 **Visible mIoU、Dice、Boundary F、CR@50、SCR@50、全分块覆盖成功率**。耗时、显存和 P/R 单列。

| 划分 | 实验案例 / 图对数 | 引用的不同 RGB 数 |
|---|---:|---:|
| train | 6,224 | 6,778 |
| val | 2,215 | 2,371 |
| test | 2,008 | 2,283 |
| 合计 | 10,447 | 11,432 |

保留既定划分，跨划分场景与图片交集为 0；不声称 object ID 完全互斥。输入为原始 RGB 和独立二值掩码，人工查看的渲染图不作为网络输入。

## 目录

```text
scripts/          现用数据适配、训练、推理和统一指标
configs/          各模型/专家的服务器配置
docs/             实验协议、命令、结果路径及复现边界
vendor/           固定版本官方源码快照，含许可证
external/         实际使用的 DINOv2、DINOv3、MMEngine 源码快照
environments/     四套独立 Conda/Pip 规格与安装记录
data_tools/       分块统计、生成与检查脚本
tests/            已有回归检查；部分需要恢复数据后运行
patches/          已有 VGGT 特征返回补丁
metadata/         官方版本、文件校验与数据版本记录
tools/            同步备份、检查与恢复工具
```

官方演示图片、视频、PDF、Notebook 和 Git 历史未复制。模型源码作为普通文件保存，直接浏览和下载可用；官方版本与补丁记录在 [sources.lock.json](metadata/sources.lock.json)，第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 阅读与使用

1. [配置总览](docs/00_阅读入口与配置总览.md)
2. [训练、验证命令与结果位置](docs/07_终端命令与结果查看.md)
3. [备份、恢复与后续同步](docs/BACKUP_AND_RESTORE.md)

**当前训练工作区仍是 `/root/datasets/MUVA_experiments`，本仓库位于 `/root/repos/MUVA-VGGTS-EXAM`。** 配置与入口保留现用绝对路径，备份副本不能直接当作已配置好的新训练目录；新服务器应按恢复说明重建环境、数据、权重路径及官方 Git 校验状态。

V²-SAM 完整 Multi+PCCS 评估需要 Fusion 和 Visual 两个专家分别训练完成；Fusion 的训练期间验证不能代替完整 Multi 的最终评估。

## 更新备份

在现服务器中，先同步明确列入备份的代码，再检查、提交和推送：

```bash
cd /root/repos/MUVA-VGGTS-EXAM
/opt/conda/envs/vggts/bin/python tools/sync_from_server.py --apply
/opt/conda/envs/vggts/bin/python tools/verify_backup.py
git diff --stat
git add .
git commit -m "Update MUVA experiment code backup"
git push origin main
```

同步脚本只写这个独立备份目录，不修改原训练工程、不启动训练。代码检查不等于完整训练或测试通过；训练效果应查看正式运行结果。
