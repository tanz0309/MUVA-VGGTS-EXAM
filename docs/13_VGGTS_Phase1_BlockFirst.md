# VGGT-S 第一阶段：目标侧 Block-First 点选择

日期：2026-11-19。范围：仅替换裁剪后 40→5 的目标提示点选择，固定 5 点预算；不训练、不改网络、不改缓存 recipe、不改六项指标。

## 1. 版本与回滚

| 项 | 值 |
| --- | --- |
| 改造前基线 tag | `pre-phase1-blockfirst`（指向 `92a6108`） |
| 开发分支 | `phase1-block-first` |
| 本阶段版本 | `block_first_v1` |

回滚方式（任选其一）：

```bash
# 代码回到改造前
git checkout pre-phase1-blockfirst -- .
# 或切回基线版本
git switch --detach pre-phase1-blockfirst
```

已有训练权重、数据缓存和结果目录不受影响；本次只改推理选择逻辑。

## 2. 改动清单

| 文件 | 动作 | 内容 |
| --- | --- | --- |
| `scripts/vggts_point_selector.py` | 新增 | 配置校验、稳定去重、DBSCAN、Medoid、容量约束配额、簇内 FPS、KMeans 回退与审计 |
| `scripts/vggts_official_data.py` | 修改 | `MuvaPreprocessing.__init__` 增加 `selector_config`；`prepare()` 替换 `cluster_points()` 调用；`audit` 增加 `point_selector` 与 `selection_inputs` |
| `scripts/run_vggts_official.py` | 修改 | 提取 `build_parser()`、新增 `validate_run_arguments()`、`--selector-config`、`smoke-evaluate`；`run_config.json` 增加来源与可比性字段 |
| `scripts/compare_vggts_selectors.py` | 新增 | A/B 结果比较：可比性校验、六项指标差值、分组与回退统计 |
| `configs/point_selectors/kmeans.json` | 新增 | 显式复现基线（省略参数时默认仍是 KMeans） |
| `configs/point_selectors/block_first_v1.json` | 新增 | `eps=0.05`、`min_samples=2`、`budget=5` |
| `run.sh` | 修改 | 帮助文本补充 `smoke-evaluate` 与 selector 用法 |
| `tests/test_vggts_point_selector.py` | 新增 | 算法、配置、回退、审计测试 |
| `tests/test_vggts_phase1_cli.py` | 新增 | 参数关系测试 |
| `tests/test_compare_vggts_selectors.py` | 新增 | 比较器可比性与差值测试 |
| `tests/test_vggts_official_data.py` | 修改 | 修复过期 batch 断言（2→当前 8），新增 selector 集成与回退测试 |

未改动：基础 config、缓存 recipe、源点采样、网络、裁剪、细化、六项指标、官方 `vendor/` 与 `external/` 源码。

## 3. 算法实现

默认参数：`budget=5`、`eps=0.05`、`min_samples=2`。`eps` 在 518 画布上约 25.9 像素，是待验证起点，不是最优参数。

- **距离归一化**：唯一候选点坐标除以 `image_size`（518，不除以 517）。
- **稳定去重**：完全相同的 float32 `(x,y)` 去重，保留首次出现的原始索引；不合并近邻点。审计同时保留原始/唯一映射。
- **空间簇**：`DBSCAN(eps, min_samples, metric='euclidean', algorithm='brute', n_jobs=1)`；噪声不救援、不补预算；非噪声簇按簇内最小原始索引重新编号。
- **超预算**：非噪声簇数 > 5 时，按唯一候选数降序、簇内最小原始索引升序保留 5 簇，其余记录为舍弃。
- **Medoid**：簇内欧氏距离和最小的真实候选；距离和并列时取原始候选索引较小者。不是“均值中心最近点”。
- **配额**：每簇保底 1，剩余按容量约束最大余数法分配；固定算例 20/12/8 → 2/2/1。
- **簇内 FPS**：以 Medoid 为起点，每次选到已选点最小距离最大的簇内候选；不跨簇抢预算，不重复选点。
- **回退**：唯一坐标 < 5、全为噪声、保留簇总容量 < 5 时，在**未去重的原始 40 点**上调用官方 `cluster_points(points, K=5)`；审计记录 `fallback_reason`，`selected_indices=null`。
- **失败策略**：输入 shape/NaN/越界/参数错误显式抛错，不被回退吞掉。

Block-First 成功时 5 个提示点均来自真实候选且互不重复；回退可能产生均值中心或重复中心，审计用 `method_effective` 区分。

## 4. 输出与审计

每次评估仍产生原有文件（`summary.*`、`per_case.*`、`predictions/`、`prompts/`、`status.json` 等）。新增：

- `prompts/<case>.json::point_selector`：方法、参数、候选、簇、配额、Medoid、噪声、回退原因、选择耗时。
- `prompts/<case>.json::selection_inputs`：原始 40 点、目标裁剪 RGB、参考裁剪 RGB/Mask、裁剪 meta 的 SHA-256，用于 A/B 逐例可比性核验。
- `run_config.json`：`point_selector`、`evaluation_checkpoint`（含文件 SHA-256）、`metric_implementation_sha256`、`evaluation_subset`、`point_cache`、`runtime_versions`。

## 5. 命令

### 5.1 单元与回归测试

```bash
cd /root/repos/MUVA-VGGTS-EXAM

PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -m unittest discover \
  -s tests -p 'test_vggts_point_selector.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -m unittest discover \
  -s tests -p 'test_vggts_phase1_cli.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -m unittest discover \
  -s tests -p 'test_compare_vggts_selectors.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -m unittest discover \
  -s tests -p 'test_metrics.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -m unittest discover \
  -s tests -p 'test_vggts_official_data.py' -v
/opt/conda/envs/vggts/bin/python tools/verify_backup.py
```

### 5.2 8 例管线验证（用户启动）

```bash
cd /root/repos/MUVA-VGGTS-EXAM
VGGTS_PHASE1_CKPT=/root/repos/MUVA-VGGTS-EXAM/outputs/vggts_official/train/muva_seed42_run01/best.pth

GPU_ID=0 bash run.sh smoke-evaluate vggts --split val \
  --checkpoint "$VGGTS_PHASE1_CKPT" --limit 8 \
  --output outputs/vggts_official/phase1/smoke_kmeans_run01

GPU_ID=0 bash run.sh smoke-evaluate vggts --split val \
  --checkpoint "$VGGTS_PHASE1_CKPT" \
  --selector-config configs/point_selectors/block_first_v1.json --limit 8 \
  --output outputs/vggts_official/phase1/smoke_block_first_run01
```

### 5.3 完整 val 对照（用户启动）

```bash
GPU_ID=0 bash run.sh evaluate vggts --split val \
  --checkpoint "$VGGTS_PHASE1_CKPT" \
  --output outputs/vggts_official/phase1/val_kmeans_run01

GPU_ID=0 bash run.sh evaluate vggts --split val \
  --checkpoint "$VGGTS_PHASE1_CKPT" \
  --selector-config configs/point_selectors/block_first_v1.json \
  --output outputs/vggts_official/phase1/val_block_first_run01

/opt/conda/envs/vggts/bin/python scripts/compare_vggts_selectors.py \
  --baseline outputs/vggts_official/phase1/val_kmeans_run01 \
  --candidate outputs/vggts_official/phase1/val_block_first_run01 \
  --output outputs/vggts_official/phase1/compare_val_run01
```

比较器输出 `comparison.{csv,json,md}`、`paired_deltas.csv`、`selector_summary.json`；六项主指标按全部样本宏平均，失败样本保留在分母。

### 5.4 小规模比较

```bash
/opt/conda/envs/vggts/bin/python scripts/compare_vggts_selectors.py \
  --baseline outputs/vggts_official/phase1/smoke_kmeans_run01 \
  --candidate outputs/vggts_official/phase1/smoke_block_first_run01 \
  --allow-smoke --output outputs/vggts_official/phase1/compare_smoke_run01
```

8 例结果只验证管线与可比性，不能当作研究结论，也不能据此反复挑 `eps`。

## 6. 测试与验证记录

2026-11-19，CPU 测试全部通过：

| 测试 | 结果 |
| --- | --- |
| `test_vggts_point_selector.py` | 29 项通过 |
| `test_vggts_phase1_cli.py` | 11 项通过 |
| `test_compare_vggts_selectors.py` | 12 项通过 |
| `test_metrics.py` | 7 项通过 |
| `test_vggts_official_data.py` | 17 项通过 |
| `tools/verify_backup.py` | 2164 份官方/依赖源码一致，0 错误 |

兼容性证据：默认不传 selector 时提示点与 `cluster_points(points, K=5)` 逐位一致；同一缓存下 A/B 的 source、source_mask、target、tracks、inliers、meta 完全相同；recipe 与 cache_path 不变；两种模式均不读取目标 GT；回退在原始 40 点上执行。

**未执行（由用户启动）**：8 例 GPU 管线验证、完整 val、test。本文件不预填任何提升数字。

过期断言修复：`test_training_keeps_author_refinement_and_loss_hyperparameters` 原断言 `train_batch_size==2` 与当前有效配置（8）漂移，改为对照 `config['training_batch_size']`，未修改训练配置。

## 7. 已知限制

- 空间簇不是 GT 可见分块，可能合并、分裂或错位；代码与日志统一称 cluster。
- 固定 5 点预算下，超预算会舍弃小簇；不保证覆盖全部真实分块。
- 首阶段 50 点传播、删除 10 点和裁剪不在本阶段范围内；缓存与裁剪已丢失的分块无法由 selector 恢复。
- 新提示分布对当前 checkpoint 是否友好，需要完整 val 的六项指标判断。
- `eps=0.05`、`min_samples=2` 是预先固定的起点；如需搜索，只能在 val 上使用有限组合，并在报告中保留全部运行；test 不能参与选参。

## 8. 结果解释边界

- 覆盖类指标提升可以支持“最终 Mask 的分块面积恢复改善”，不能推出每块获得独立身份。
- 覆盖高但 IoU/Dice 下降，可能同时存在漏块恢复与误分，必须结合 Precision 与 Boundary F 解读。
- 未提升只说明当前参数/策略/权重下未观察到收益，不能断言所有分块优先方法不可行。
