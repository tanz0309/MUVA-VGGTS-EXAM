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

2026-11-24 12:39 UTC：99项单元测试通过；GPU smoke和300例实验尚待执行。完成后追加实际结果，不将上述命令当作已执行证据。

完成A组后按用户方案暂停，不自动实施B/C或训练。
