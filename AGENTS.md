# MUVA 统一实验项目维护约定

- 用户于 2026-11-18 指定 `/root/repos/MUVA-VGGTS-EXAM` 为后续实验的唯一代码、配置、文档和命令入口。Codex 与 DeepSeek 均直接在本项目维护并通过 Git 管理。
- `/root/datasets/MUVA_experiments` 保留历史运行和固定官方依赖，已有训练继续使用原路径。不要停止、重启、移动或覆盖它们；通过本项目忽略的运行目录链接复用数据和结果。
- 本机协作记录遵循 `/root/AGENTS.md`，接手先读 `/root/docs`；在其他服务器使用时，以当地用户指令为准，不假设这些目录存在。
- 使用对应 Conda Python。四种方法分别使用独立环境；V²-SAM 的专家共享其方法环境。
- 后续修改以本项目为准，禁止从旧工程同步覆盖本项目；旧 `sync_from_server.py` 已退役。运行链接用 `tools/link_runtime.py` 管理。参数和官方依赖路径保持现有 checkpoint 兼容；变更需记录。
- `vendor/` 和 `external/` 保存官方/依赖源码快照。不得把修改后的网络当作未改官方模型。已有补丁和来源必须记录在 `patches/`、`metadata/sources.lock.json`。
- 不提交数据、掩码、权重、运行输出、环境目录或凭据。新增备份路径时同步维护检查工具和恢复说明。
- seed 42、既定划分及六项主指标保持一致；test 不用于选择模型或调整阈值。
- 上传前运行 `tools/verify_backup.py`，核对不可变官方源码及差异，确认运行目录和凭据未被跟踪；禁止强制推送或覆盖他人提交。
