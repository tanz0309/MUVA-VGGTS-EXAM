# MUVA 四模型对比实验

INSID3、O-MaMa、VGGT-S、V²-SAM 在 MUVA 分块目标子集上的统一实验项目。

**2026-11-18 起，`/root/repos/MUVA-VGGTS-EXAM` 是 Codex、DeepSeek 后续实验的统一依据。** 代码、配置、文档与新实验命令都在本项目维护，以 Git 管理。原目录保留已有运行和固定依赖，数据、权重、环境及运行产物不提交 GitHub。

## 当前实验配置

| 方法 | 入口 | 独立 Conda 环境 | batch | epochs |
|---|---|---|---:|---:|
| INSID3 | `insid3` | `insid3-muva` | 1 | 免训练 |
| O-MaMa | `omama` | `omama` | 24 | 3 |
| VGGT-S | `vggts` | `vggts` | 8 | 12 |
| V²-SAM Fusion | `v2sam` | `v2sam` | 16 | 24 |
| V²-SAM Visual | `v2sam-visual` | 同上 | 16 | 12 |

公共 seed **42**；作者内部固定 KMeans seed 0 保留。六项主指标为 **Visible mIoU、Dice、Boundary F、CR@50、SCR@50、全分块覆盖成功率**。既定 train/val/test 为 **6224/2215/2008** 案例，共 **10447** 图对；不同 RGB 共 **11432** 张。场景、图像跨划分交集为 0，不声称 object ID 完全互斥。原始 RGB 与独立掩码作为输入。

官方网络结构保持原方案；既有适配、兼容处理及 PCCS 补齐边界见 [复现说明](docs/10_论文核对与复现边界.md)。V²-SAM Multi+PCCS 需要 Fusion 与 Visual 两个训练专家，单专家验证不能代替最终 Multi 评估。

## 从统一项目运行

```bash
cd /root/repos/MUVA-VGGTS-EXAM
/opt/conda/envs/vggts/bin/python tools/link_runtime.py --apply
bash run.sh help
```

本机 `data/cache/logs/outputs/weights` 为被 Git 忽略的共享存储链接。现有训练和权重可从本项目访问；已经启动的任务继续使用原路径，不重启或移动。

配置保留固定官方依赖、Conda 和预训练权重路径，兼容既有 checkpoint 的严格配置校验。官方独立 Git 工作区作为只读依赖复用，项目 `vendor/`、`external/` 保存对应源码快照。后续项目代码与配置直接在本仓库修改，禁止从旧工程覆盖同步。

- [配置总览](docs/00_阅读入口与配置总览.md)
- [项目架构与模型代码导读](docs/12_项目架构与模型代码导读.md)
- [VGGT-S 第一阶段 Block-First 选点](docs/13_VGGTS_Phase1_BlockFirst.md)
- [训练、验证命令和结果位置](docs/07_终端命令与结果查看.md)
- [统一目录及已有实验衔接](docs/11_统一项目目录与现有实验衔接.md)
- [版本管理和新服务器恢复](docs/BACKUP_AND_RESTORE.md)

## 目录

`scripts/` 为适配、训练和指标；`configs/` 为配置；`docs/` 为协议与命令；`vendor/`、`external/` 为固定源码；`environments/` 为四套环境规格；`data_tools/` 为数据处理/恢复脚本；`tests/` 为已有检查；`patches/`、`metadata/` 保留补丁、来源及数据版本；`tools/` 为运行链接、检查和恢复工具。

第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。初始导入摘要作历史依据，后续项目改动由 Git 管理；官方/依赖快照继续严格校验。

## 提交更新

```bash
cd /root/repos/MUVA-VGGTS-EXAM
/opt/conda/envs/vggts/bin/python tools/verify_backup.py
git diff --stat
git add .
/opt/conda/envs/vggts/bin/python tools/verify_backup.py --staged
git commit -m "Describe the experiment change"
git push origin main
```

旧 `tools/sync_from_server.py` 已移除，后续直接维护本项目。完整训练和评估按用户指令启动；操作记录写入 `/root/docs`，接手先读双方最新交接。
