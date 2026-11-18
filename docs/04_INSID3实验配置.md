# INSID3：免训练可见分割

## 模型配置

使用固定提交 `0c165a10cf52ab91f335883d06260de86854adbe` 的官方 INSID3 类（源码在 `vendor/INSID3-official`）和 DINOv3 ViT-L/16 权重。输入边长 1024、位置去偏 SVD 分量 500、聚类 tau=0.6、合并阈值 0.2；掩码双线性恢复到原始目标图尺寸，不启用可选 CRF。配置文件为 `configs/insid3.json`。

每次都传入一张参考图及其真实可见掩码、一张原始目标图。模型每对重置参考状态，不让不同案例累积成多参考输入。各对独立执行，不跨对复用特征，因此计时包括该对参考特征计算。

## 不训练

不提供 INSID3 训练命令；运行器会拒绝其 train / smoke-train 请求。train 划分仍保留在公共数据中供其他需要训练的方法使用。

## 小规模检查

从 val 中固定取两个案例：第一个，以及另一场景中一个至少有三块的案例。先用第一个做一次预热，再记录两个正式计时输出。这只是接口检查，不是验证集性能估计，也不据此调参。

## 正式启动

```bash
# 重做小规模检查
GPU_ID=0 bash /root/datasets/MUVA_experiments/run.sh smoke insid3

# 完整验证集：2215 个案例，用户自行启动
GPU_ID=0 bash /root/datasets/MUVA_experiments/run.sh evaluate insid3 --split val

# 完整测试集：2008 个案例，冻结方案后用户自行启动
GPU_ID=0 bash /root/datasets/MUVA_experiments/run.sh evaluate insid3 --split test
```

输出在 `outputs/insid3/<mode>/` 下自动创建时间戳目录，防止覆盖既有结果。包含逐例二值预测 PNG、逐例 CSV/JSONL、六指标及辅助指标汇总、运行配置和完成状态。
