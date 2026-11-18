# 代码备份维护约定

- 这是 `/root/datasets/MUVA_experiments` 的独立代码备份。当前服务器上的训练仍在原工程运行；不要为整理本仓库而停止、重启或改动正式训练。
- 本机协作记录遵循 `/root/AGENTS.md`，接手先读 `/root/docs`；在其他服务器使用时，以当地用户指令为准，不假设这些目录存在。
- 使用对应 Conda Python。四种方法分别使用独立环境；V²-SAM 的专家共享其方法环境。
- `scripts/`、`configs/`、`docs/00` 至 `docs/10` 以原工程为来源；通过 `tools/sync_from_server.py` 同步。不要只改备份而误以为已改变运行中的工程。
- `vendor/` 和 `external/` 保存官方/依赖源码快照。不得把修改后的网络当作未改官方模型。已有补丁和来源必须记录在 `patches/`、`metadata/sources.lock.json`。
- 不提交数据、掩码、权重、运行输出、环境目录或凭据。新增备份路径时同步维护检查工具和恢复说明。
- seed 42、既定划分及六项主指标保持一致；test 不用于选择模型或调整阈值。
- 上传前运行 `tools/verify_backup.py`，检查差异并确认没有意外文件；禁止强制推送或覆盖他人提交。
