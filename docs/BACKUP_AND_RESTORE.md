# 项目版本管理与新服务器恢复

2026-11-18 起，用户指定 `/root/repos/MUVA-VGGTS-EXAM` 为统一实验项目。此前“只作备份、从旧工程同步”的方式已经退役；直接在此维护代码、配置和文档。

## 当前服务器

在本项目用 `tools/link_runtime.py --apply` 关联现有运行存储，随后从本项目 `run.sh` 启动任务。工具只创建被 Git 忽略的链接，不搬动文件。已有任务原位运行，完整映射见 [统一目录说明](11_统一项目目录与现有实验衔接.md)。

修改后执行 `tools/verify_backup.py`、检查 Git 差异、暂存，再用 `tools/verify_backup.py --staged` 检查并提交推送；命令见 README。禁止再从旧工程覆盖同步，不强制推送或覆盖其他 Agent 的提交。

`metadata/source_files.json` 是首次导入的历史摘要。项目脚本、配置和文档的后续合法修改由 Git 跟踪，检查工具不要求回退到旧工程；固定 `vendor/`、`external/` 仍严格校验。`--compare-live` 仅额外核对固定官方/依赖原件，不再将旧工程当作项目代码真源。

## 新服务器恢复

### 1. 获取代码与环境

```bash
git clone https://github.com/tanz0309/MUVA-VGGTS-EXAM.git /root/repos/MUVA-VGGTS-EXAM
cd /root/repos/MUVA-VGGTS-EXAM
```

`environments/<name>/` 保存四套 Conda 精确列表、YAML、PyTorch/Pip 需求及 editable 安装记录。先按 conda-only 规格建环境，再恢复对应依赖，不混装四种方法。

环境二进制不在 GitHub。V²-SAM 的 MMCV 2.1.0 需要匹配 torch/CUDA 编译，现服务器使用 CUDA 12.4、架构 8.9；MMEngine 为 `external/mmengine-runtime` 对应源码。需求文件里的本地 Decord wheel、旧 editable 路径是历史安装记录，必须恢复材料或重建，不假设一条 pip 命令可全部恢复。

### 2. 固定官方依赖

配置保留官方独立 Git 工作区路径，以保持已训练 checkpoint 和源码检查兼容。`vendor/` 是普通文件快照，不能用外层 Git 提交伪装成作者版本。

```bash
/path/to/conda/envs/vggts/bin/python tools/restore_vendor.py \
  --destination /root/datasets/MUVA_experiments/vendor
```

仅在新/空目录恢复六套锁定上游、已有 VGGT 特征返回补丁和 V²-SAM 兼容别名。外部 DINOv2/DINOv3、MMEngine 从 `external/` 按配置及安装记录恢复。基础权重另行下载，保留官方网络结构。

### 3. 数据和运行存储

数据在私有 Hugging Face 仓库 `tzz2002/MUVA`，固定版本见 `metadata/dataset.json`：

```bash
/path/to/conda/envs/insid3-muva/bin/hf auth login
/path/to/conda/envs/insid3-muva/bin/hf download tzz2002/MUVA --repo-type dataset \
  --revision 5c027b9917fe0373ca80f5b5dea5c317edf34603 \
  --include 'core-20261109/*' --local-dir /root/muva-download
/path/to/conda/envs/insid3-muva/bin/python data_tools/restore_hf_data.py \
  --download-dir /root/muva-download/core-20261109 \
  --destination /root/muva-restored-datasets
```

恢复工具读取下载包 manifest，验证四个归档、安全解压并核查案例。合并回 `/root/datasets` 时不覆盖项目代码。清单有原绝对路径，换存储位置需统一迁移和复核。

准备共享 `data/cache/logs/outputs/weights` 后，在项目运行 `tools/link_runtime.py --apply`；目标必须存在，工具不覆盖已有项目路径。任务权重和结果不在 GitHub，需要从独立运行备份恢复。

### 4. 检查后运行

```bash
/path/to/conda/envs/vggts/bin/python tools/verify_backup.py
bash run.sh help
```

核对环境、固定依赖、权重、清单及 checkpoint 配置，再按 [命令文档](07_终端命令与结果查看.md) 运行。文档 06 与初始 `metadata/validation.json` 是历史证据，不能替代新服务器 GPU 检查。

旧 `configs/vggts.json` 保留作历史，统一入口使用 `vggts_official.json`。部分数据依赖检查有历史参数断言，未运行的检查不宣称通过。seed 42 不保证所有 CUDA 算子逐位确定。
