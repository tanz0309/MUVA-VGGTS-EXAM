# 代码备份、恢复与后续管理

## 当前工作区与备份边界

现用工程为 `/root/datasets/MUVA_experiments`，GitHub 工作区为 `/root/repos/MUVA-VGGTS-EXAM`。备份工具从现用工程复制允许的源码，不移动文件、不改变正在训练的进程，也不修改官方网络结构。

备份包含现用适配/训练代码、八份配置、统一协议文档、现有回归检查、六套官方源码快照、实际使用的三套外部源码、四套环境规格、分块统计/恢复脚本、源码来源和已有补丁。按文件记录原始路径及 SHA256；Markdown 中指向工程内部的文件链接转为仓库相对链接，其他程序代码按字节保留。

不包含原始 RGB、标注、掩码、案例数据清单、预训练和任务权重、预测结果、日志、点缓存、环境实体、Git 历史、演示媒体、论文全文、Notebook 或任何凭据。原始服务器路径是恢复参考，不代表 GitHub 已保存这些文件。

## 在原服务器更新备份

```bash
cd /root/repos/MUVA-VGGTS-EXAM
/opt/conda/envs/vggts/bin/python tools/sync_from_server.py
/opt/conda/envs/vggts/bin/python tools/sync_from_server.py --apply
/opt/conda/envs/vggts/bin/python tools/verify_backup.py --compare-live
git diff --stat
git status --short
git add .
/opt/conda/envs/vggts/bin/python tools/verify_backup.py --compare-live --staged
git commit -m "Update MUVA experiment code backup"
git push origin main
```

默认不加 `--apply` 只显示计划。同步会重建本仓库的 `scripts`、`configs`、`tests`、`vendor`、`external`、`environments`、`data_tools`、`patches` 和 `metadata` 等来源目录；自写维护工具保存在 `tools/`，补充备份说明保存在本文。不要在来源目录手改后再运行同步而丢失自己的修改，应先修改现用工程或将新代码放在独立维护目录。

多个维护者修改时，先检查远端和本地差异，合并后再推送。不要强制推送，不自动覆盖其他人的提交。每次备份的时间和文件摘要在 `metadata/` 中。

## 在新服务器恢复

先获取仓库和校验快照，Python 使用对应 Conda 环境：

```bash
git clone https://github.com/tanz0309/MUVA-VGGTS-EXAM.git
cd MUVA-VGGTS-EXAM
/path/to/conda/envs/vggts/bin/python tools/verify_backup.py
```

### 1. 恢复四套环境

`environments/<name>/` 保存 Conda 精确包列表、YAML、分开的 PyTorch/Pip 需求和 editable 安装记录。建议先用 `environment-conda-only.yml` 建立环境，再按 `requirements-torch.txt` 和 `requirements-rest.txt` 安装对应包；不要将四种方法混装到同一个环境。

这些规格来自当前已恢复的服务器，但不包含二进制环境或编译产物。V²-SAM 的 MMCV 2.1.0 需要匹配 torch/CUDA 编译，当前服务器曾用 CUDA 12.4、架构 8.9。其 MMEngine 为 `external/mmengine-runtime` 的源码副本，以 editable 方式安装。`requirements-rest.txt` 中的本地 Decord wheel 和旧 editable 源路径是记录，GitHub 未包含该 wheel，需要从原安装材料恢复或重新构建，不能直接假设一条 pip 命令就可完整恢复。

各环境实际路径、torch 版本和复现边界见 [环境说明](03_模型范围与Conda环境.md)。恢复仅在新工作区进行，不在已运行环境上重建。

### 2. 恢复官方源码及 Git 校验状态

`vendor/` 中是普通文件快照。现用的 `scripts/source_audit.py` 要求每个主模型目录拥有匹配提交的独立 Git 元数据；直接在备份树内运行会读到外层仓库版本，不能通过该校验。

在一个全新的工作目录恢复六个固定上游版本：

```bash
/path/to/conda/envs/vggts/bin/python tools/restore_vendor.py \
  --destination /root/datasets/MUVA_experiments/vendor
```

命令在目标目录不存在或为空时才执行。它从锁定提交恢复完整官方 Git 工作区，应用已有 VGGT 特征返回补丁，并恢复 V²-SAM 的 `projects/v2sam -> v2sam_fusion` 别名。不把备份快照中的源码改写成另一个提交，不允许覆盖运行中的 vendor。

将 `scripts/`、`configs/`、`run.sh` 等复制到新工程，并按配置将 `external/dinov2-runtime`、`external/dinov3-runtime`、`external/mmengine-runtime` 放到相应源码路径。原路径缺失时，可以修改本地配置中的绝对路径，但应记录变更、保持模型结构和实验协议不变。

### 3. 恢复数据

数据单独保存在私有 Hugging Face 仓库 `tzz2002/MUVA`，固定版本见 `metadata/dataset.json`。用有权限的账号在终端登录：

```bash
/path/to/conda/envs/insid3-muva/bin/hf auth login
/path/to/conda/envs/insid3-muva/bin/hf download tzz2002/MUVA --repo-type dataset \
  --revision 5c027b9917fe0373ca80f5b5dea5c317edf34603 \
  --include 'core-20261109/*' --local-dir /root/muva-download
/path/to/conda/envs/insid3-muva/bin/python data_tools/restore_hf_data.py \
  --download-dir /root/muva-download/core-20261109 \
  --destination /root/muva-restored-datasets
```

恢复工具读取下载包内的 `manifest.json`，校验四个归档后解压到新/空目录，并检查清单输入。其输出包含 `/root/datasets` 所需的数据目录；合并回新服务器目标位置时保留已有代码，不用数据包替换整个实验工程。原始清单绝对路径以 `/root/datasets` 为基准；换位置必须统一迁移路径并复核划分。

`metadata/dataset.json` 保存既定三份清单的 SHA256，用于确认恢复后没有变更划分。GitHub 不公开这三份清单本身。

### 4. 恢复基础权重及训练结果

基础权重按各配置、上游 README 和 `docs/` 说明下载/恢复。`scripts/download_sam2.py` 保留现有 SAM2 下载入口。所有权重均未上传 GitHub，正在训练的 `best.pth`、`last.pth` 和验证输出也不属于本次代码备份；需要另行备份时单独安排存储。

核对环境、源码、输入和权重后，再按 [终端命令](07_终端命令与结果查看.md) 运行。现有 GPU 检查文档是当时服务器的历史证据，不能替代新服务器的实际运行验证。

## 已知适配与限制

- 官方主模型结构保持。已有 VGGT track head 特征返回补丁保存在 `patches/vggt-official.patch`；本地输入 clone 等兼容处理保存在适配脚本。
- O-MaMa 使用作者模块和损失，本地入口组织真实 batch；V²-SAM 使用官方专家，Multi 的 PCCS 是明确披露的论文式补齐实现。细节见文档 08～10。
- `configs/vggts.json` 为保留的旧配置，当前 `run.sh train vggts` 使用 `vggts_official.json` 和 `run_vggts_official.py`；不要混用旧权重。
- `tests/test_vggts_official_data.py` 等检查需要完整数据和既定路径；已有文件可能仍含历史参数断言。此次备份优先保留现用代码，不把未运行的检查宣称通过。
- CUDA `cumsum`、Flash Attention 等的非确定性警告不会因为固定 seed 就消失。
