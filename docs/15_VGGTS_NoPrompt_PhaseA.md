# VGGT-S 无提示分块恢复：Phase 0 / A

用户执行方案原文见 [Implementation/VGGT-S_NoPrompt_Fragment_Recovery_Ablation_Plan.md](Implementation/VGGT-S_NoPrompt_Fragment_Recovery_Ablation_Plan.md)。执行前版本 `01be52a` 已推送 main；新增实现维护在 `experiment/vggts-pass-analysis`。

## 实现范围

只实现 Phase A。正式 checkpoint、VGGT backbone、KMeans、裁剪、源点采样、六项指标与 vendor 文件均保持原样。每例先在未修改的 `OfficialEvaluator.predict()` 上临时记录其 prepared inputs；hook 在 `finally` 中移除。随后独立执行 A0/A1/A2/A3，四组共用实际正式调用的同一份特征、点和源Mask；源点特征有 clone 保护，输入内容用 SHA256 核对。独立 A1 必须与正式最终预测逐像素一致，Pass-0 low logits 也必须与正式 feedback 指纹一致，否则立即失败。

| 组别 | 目标 dense feedback | 含义 |
|---|---|---|
| A0 | None | 初次预测 |
| A1 | A0 low logits | 原生 refinement |
| A2 | None | 重复初次预测，检查确定性 |
| A3 | A0 low logits 两个空间维固定循环平移半幅 | 精确保留数值/面积的空间反馈控制 |

为核对正式路径，每例实际有正式路径2次加独立实验4次 Predictor 前向，但VGGT特征仅生成一次。这是验证开销，不能把本实验耗时与正常2次前向吞吐混为一谈。A2不是额外机制；A3也不是所有损坏反馈的代表。

所有样本推理和logits保存完毕、权重未变检查通过并写出 `inference_complete.json` 后，才调用独立事后分析模块读取GT。GT不参与提示、特征、前向或平移。候选点/selected prompt 支持在原图坐标中统计；使用与正式恢复同一crop metadata转换，点按floor落入像素，越界点计背景，不夹到图像边缘。

## 可复现命令

从 `/root/repos/MUVA-VGGTS-EXAM` 执行，使用独立 VGGT-S Conda：

```bash
env PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -B -m unittest discover -s tests -p 'test_*.py'

env PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -B -u scripts/run_vggts_ablation.py \
  --mode pass-analysis --split val \
  --checkpoint outputs/vggts_official/train/muva_seed42_run01/best.pth \
  --limit 8 \
  --output outputs/vggts_official/decoder_mechanism/smoke_A_20261124_run01

env PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/vggts/bin/python -B -u scripts/run_vggts_ablation.py \
  --mode pass-analysis --split val \
  --checkpoint outputs/vggts_official/train/muva_seed42_run01/best.pth \
  --limit 300 \
  --output outputs/vggts_official/decoder_mechanism/pass_A_20261124_run01
```

输出目录必须全新。默认seed42分层抽样，按manifest中已有分块数(2/3/4+)、visibility proxy分箱(0.1/0.4/0.7)、scene group覆盖不同场景，保留manifest顺序。选样不打开GT，不使用旧test结果或候选点支持筛选；无候选点样本数量在推理后的报告披露。该300例是机制子集，其均值不能冒称完整val成绩。若今后明确需要完整val，显式指定 `--limit 2215`。CLI目前只允许val和A组。

如推理完整结束而事后分析失败，可用 `--stage analysis --output <原输出目录>` 继续分析，无需重跑模型；已经complete的报告不会覆盖。

## 交付与核验

- `config.json` / `checkpoint_info.json`：提交、源代码/配置/manifest/正式权重 SHA256，解释器、GPU、dtype、干预参数；checkpoint必须通过既有scope/config/三划分manifest及strict state_dict校验。
- `sample_manifest.json`：固定样本顺序与原manifest记录。
- `pass_predictions/<case>/`：4组高/低分辨率float32 logits（精确保留BF16输出值）、crop及原图二值预测、原路径最终预测和逐轮输入指纹。
- `pass_metrics.csv`、`component_records.csv`、`recovery_transition.csv`、`sample_comparison.csv`：六项主指标、无候选点/无最终prompt恢复、宏/微平均、新恢复与退化、最小分块、面积和最近欧氏像素距离。
- `summary.json`、`diagnostic_report.md`、`visualizations/`：汇总与8面板图。分母为0明确写null。
- `logs/inference.jsonl`、`status.json`：逐例持久进度；不把inference_complete误报为analysis完成。

## 已执行结果（2026-11-24）

推理代码提交 `dab282a7acbe971da1ca1b1a4a761eefd1ccc99e`。99项单元测试通过；8例GPU smoke于13:20:41 UTC完成，300例val于13:27:45 UTC完整结束，失败0。正式模型为epoch12的best，SHA256 `21db6817469a9b64e90621ab999a904f066e6d864544c879d261fc91a33e3268`；两份权重文件及predictor内存state前后哈希均一致。300次使用已有点缓存，新生成0。

每例A1与原OfficialEvaluator最终掩码逐像素一致；A2与A0的最大高/低分辨率logit差均为0。CUDA仍有原协议的非确定性warning，本次对照为实测一致，不能外推为跨硬件逐位确定。

本次是**分层抽样300例val机制子集**，不是完整2215例val，也不是历史2008例test。共有772个显著分块，251例至少缺一个candidate支持分块；无候选点分块393个、无最终prompt分块513个。两类不能互换：native KMeans中心可落在没有实际候选点的分块内，本次有4个这样的分块；同时无candidate与无prompt的交集为389个。

| 组别 | Visible IoU | Dice | Boundary F | CR@50 | SCR@50 | 全分块覆盖@50 |
|---|---:|---:|---:|---:|---:|---:|
| A0 初次预测 | 75.5135% | 83.1838% | 89.0338% | 71.6139% | 53.4167% | 48.3333% |
| A1 原生反馈 | 76.0714% | 83.4773% | 89.0449% | 70.6639% | 52.0833% | 47.0000% |
| A2 重复None | 75.5135% | 83.1838% | 89.0338% | 71.6139% | 53.4167% | 48.3333% |
| A3 平移反馈 | 71.0776% | 79.8859% | 87.0765% | 63.9778% | 43.4167% | 36.0000% |

以下分块恢复率是分块微平均；逐样本宏平均另保存在summary.json和完整报告，不能与下表互换。

| 分组 | A0恢复/总数 | A1恢复/总数 | A3恢复/总数 |
|---|---:|---:|---:|
| 无candidate | 194/393（49.36%） | 184/393（46.82%） | 152/393（38.68%） |
| 无selected prompt | 282/513（54.97%） | 272/513（53.02%） | 231/513（45.03%） |
| candidate和selected prompt均无 | 192/389（49.36%） | 182/389（46.79%） | 151/389（38.82%） |

### 可支持的解释

1. 无点分块的大量恢复已发生在第一次预测：A0恢复282个无prompt分块，A1最终272个，其中268个是保留的初次成功，只有4个是新增恢复。因此，本子集中refinement并非无prompt分块恢复的主要新增来源；首次推理包含的源Mask、跨视角特征与Token机制仍需B/C受控消融区分。
2. 原生feedback相对A0新增恢复4个分块，同时14个退化，净减少10个。无candidate组为新恢复3、退化13；无prompt组为新恢复4、退化14。整体IoU提高0.5579pp，但全分块覆盖下降1.3333pp，不能把总体IoU改善当作分块发现能力提高。
3. A3相对A1 IoU下降4.9938pp、无prompt恢复下降7.9922pp，支持反馈空间对应有作用；这不能证明每次refinement都补回遗漏分块，也不能定位到某种attention。
4. A1平均false-positive像素从1066.53降到980.50，precision从87.74196%到88.42499%，不支持用整体Mask扩张解释本次IoU改善。局部变化仍见逐例记录与可视化。
5. 面积比例Q1分块恢复从37.8238%降到33.1606%，Q4两组均97.4093%。最近距离Q1到Q4的A0恢复反而由55.44%升至74.23%，存在面积、点支持等混杂，不能套用“越远一定越难恢复”的解释。

### 结果位置与交付完整性

服务器输出根目录：`outputs/vggts_official/decoder_mechanism/pass_A_20261124_run01/`。

- 完整报告：`diagnostic_report.md`，含六项指标、宏/微平均、分母与恢复/退化表。
- `pass_metrics.csv` 1200行；`component_records.csv` 3088行；`recovery_transition.csv` 2316行；`sample_comparison.csv` 300行。
- 300例共5400份预测/原路径/输入审计文件，约6.45GB；12张代表性8面板图。
- `artifact_verification.json` 记录产物、样本数、40/5点数、输入一致性、源文件和权重未变的核验。
- `logs/`包含控制台日志、启动argv、单测结果摘要、smoke验收状态、修改文件清单和`implementation.diff`。原始权重/数据/输出均未上传Git。
- `config.json`、`checkpoint_info.json`和`sample_manifest.json`保留本次全部版本和样本证据。

完成A组后按用户方案暂停，不自动实施B/C或训练。
