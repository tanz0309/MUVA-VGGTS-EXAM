# VGGT-S：无提示点可见分块恢复机制——深入消融实验方案

**版本：** 1.0  
**日期：** 2026-11-24  
**适用项目：** `MUVA-VGGTS-EXAM`（以 `MUVA-VGGTS-EXAM-main-11.24.zip` 为代码基准）  
**研究对象：** 处理后的 MUVA 多视角遮挡分块数据  
**实验性质：** 机制分析与推理消融；第一轮不训练、不改正式权重  
**交付对象：** 服务器端 Claude Code / Codex Agent

---

## 0. 任务定义与研究边界

### 0.1 真正要回答的研究问题

同一个物体在目标视角中可能因为手、工具或其他物体遮挡，呈现为多个空间不连续的可见分块。模型的任务不是只分割最大区域，而是识别这些可见分块属于同一个查询物体，并尽可能完整地分割全部可见部分。

本阶段聚焦一个由现有诊断结果提出的机制问题：

> 当某个真实可见分块中没有任何 VGGT 传播候选点、甚至没有最终提示点时，VGGT-S 为什么仍可能在最终 Mask 中覆盖该分块？这种恢复来自源 Mask、双视角特征交互、稀疏点提示、Mask Token，还是迭代 Mask Refinement？

本阶段**不是**继续优化 DBSCAN，也不是直接设计新 Loss。先弄清楚现有模型的恢复来源，才能有依据决定下一步应改哪一个模块。

### 0.2 已有诊断结果（作为研究动机，不作为本轮重新计算的结论）

用户提供的 2026-11-24 Block-First 诊断报告给出：

- Test 样本：2008。
- 显著 GT 分块：4235。
- 没有任何 40 个候选点落入的 GT 分块：1876（44.3%）。
- 至少缺一个分块的样本：1515（75.4%）。
- 40 点覆盖全部显著分块的样本：493（24.6%）。
- 无候选点支持的 1876 个分块中，1061 个（56.6%）仍被最终 Mask 覆盖。
- 40 点选择后，5 个提示点未覆盖全部分块的 1745 例中，981 例（56.2%）最终 Mask 仍覆盖全部分块。
- KMeans 与 Block-First 最终 Mask 全分块覆盖率均约 61.2%，提示点覆盖率变化没有转化为最终 Mask 的可测收益。

**解释边界：**

1. 56.6% 是“无候选点支持的 GT 分块”这一分块集合中的覆盖比例，不是样本成功率。
2. 61.2% 是“最终 Mask 覆盖样本内全部显著 GT 分块”的样本比例。
3. 这两个数字不能互换，也不能据此直接断言模型已经具备稳定的自主分块发现能力。
4. “无候选点”不等于“无提示信息”：模型仍有源 Mask、源/目标 VGGT 特征、其他位置的源点/目标点 Token，以及 Mask Token。
5. 56.6% 是已有最终预测的事后统计；必须通过分阶段预测与干预实验，才可能解释其成因。

### 0.3 本轮硬性边界

- 使用同一个正式 VGGT-S checkpoint。
- 使用同一个 VGGT backbone checkpoint。
- 固定验证集样本清单、数据 manifest、seed 和预处理缓存。
- 不改变正式模型权重，不进行 optimizer 更新。
- 不修改原始六项指标定义。
- 不让 GT 进入任何模型输入、Prompt、候选点选择、Attention 或 Mask Refinement。
- GT 只用于推理结束后的逐分块分析。
- 不覆盖既有实验输出。
- 第一轮先执行 A 组，再根据结果决定是否执行 B、C、D 组。
- 不在本轮 test split 上调参；使用 val split 做机制研究。最终冻结方案后再按既定协议运行 test。

---

## 1. 代码基线与需要核对的真实调用链

以下路径依据上传的 `MUVA-VGGTS-EXAM-main-11.24.zip`。Agent 开工前必须再次在服务器工作树核对函数签名和代码版本，并记录 Git commit / 文件 SHA256。若服务器代码与 ZIP 不同，以服务器实际代码为准并在报告中说明差异。

### 1.1 推理入口

**文件：** `scripts/vggts_official_data.py`  
**类/函数：** `OfficialEvaluator.predict(reference, reference_mask, target)`

当前推理主流程可概括为：

1. `self.preprocessing.prepare(...)` 准备源图、源 Mask、目标裁剪图、目标提示点和特征缓存。
2. `vggt_feat(...)` 生成源侧点 `src_points` 与双视角 VGGT 特征 `features`。
3. `sample_feat_at_points(features[:, 0], src_points, ...)` 提取源点视觉特征 `src_features`。
4. `p['target_points']` 转为目标侧提示点 Tensor。
5. `refinement = None`。
6. 循环 `range(refine_time + 1)` 调用 `Predictor.forward(...)`；每次将上一次 `low` 作为下一次 `tar_mask_b`。
7. 使用最终 `logits` 做 crop recovery、resize，返回预测 Mask。

当前 `configs/vggts_official.json` 中 `refine_time=1`，所以推理通常执行两次 Predictor：

- Pass 0：`tar_mask_b=None`。
- Pass 1：`tar_mask_b=Pass 0 的 low-resolution logits`。

**必须新增逐 Pass 记录能力。** 不能只保存最终 Mask，否则无法回答恢复发生在哪一轮。

### 1.2 Predictor

**文件：** `vendor/VGGT-S-official/src/model/predictor.py`  
**函数：** `Predictor.forward(...)`

输入包括：

- `feat_map_b`：源/目标 VGGT 特征，形状注释为 `[B, 2, 128, 518, 518]`。
- `src_points_b`：源点坐标。
- `tar_points_b`：目标点坐标。
- `src_mask_b`：源 Mask。
- `src_points_feat_b`：从源视角 VGGT 特征图采样的源点特征。
- `tar_mask_b`：迭代 refinement 的目标 Mask 提示；首轮为 `None`。

`Predictor.forward()` 将源 Mask 转为强 logits（前景 +30、背景 -30），调用 PromptEncoder，之后调用 MaskDecoder，输出高分辨率 logits 和 low-resolution logits。

### 1.3 PromptEncoder

**文件：** `vendor/VGGT-S-official/src/model/prompt_encoder.py`  
**函数：** `PromptEncoder.forward(...)`

当前实现的稀疏 Embedding 拼接顺序是：

```python
sparse_embeddings_b = torch.cat(
    [src_points_embed, tar_points_embed, src_points_feat_b],
    dim=1
)
```

因此 sparse token 包含三组：

1. 源点坐标 Embedding。
2. 目标点坐标 Embedding。
3. 源点视觉特征 Embedding。

密集提示包括：

- `src_dense_embeddings_img_b = mask_downscaling(src_mask_b)`。
- 若 `tar_mask_b is not None`，则 `tar_dense_embeddings_img_b = mask_downscaling(tar_mask_b)`；否则为 `None`。

注意：PromptEncoder 会对 `src_points_feat_b` 原地加上源点类型 Embedding。项目中的 `protect_feature_inputs()` 已注册 forward pre-hook，对该输入执行 `.clone()`，避免重复前向时累积原地修改。消融实现必须保留这一保护，或确保每次调用都传入独立 clone。

### 1.4 MaskDecoder

**文件：** `vendor/VGGT-S-official/src/model/mask_decoder.py`  
**函数：** `MaskDecoder.forward(...)`

已核对的结构事实：

- `num_mask_tokens = 1`，当前只有一个可学习 Mask Token。
- Mask Token 与 sparse prompt embeddings 拼接后送入 Transformer。
- `feat_map_b[:, 0]` 和 `feat_map_b[:, 1]` 分别是源/目标特征。
- 源/目标图像特征沿宽度维拼接，进入 TwoWayTransformer。
- Transformer 输出的 Mask Token 经 Hypernetwork MLP 生成动态权重。
- 动态权重与**目标侧更新后的 image feature map**做像素位置点积，生成 Mask logits。

因此，Mask Token 不是一个只在点附近预测局部区域的输出头；它可以通过 Transformer 从全局图像 Token 中获取信息，再对目标特征图产生全图 Mask。但“它是否因此恢复无提示分块”仍是待验证假设。

### 1.5 TwoWayTransformer

**文件：** `vendor/VGGT-S-official/src/model/transformer.py`  
**类/函数：** `TwoWayTransformer.forward()`、`TwoWayAttentionBlock.forward()`

已核对的交互结构：

1. 图像 Token 先由源/目标特征沿宽度拼接得到。
2. 每个 `TwoWayAttentionBlock` 将图像 Token 恢复空间布局，分成 source / target 两部分。
3. 第一层（`skip_first_layer_pe=True`）将源 Mask dense embedding 加到 source image feature；若 target mask embedding 不为 None，则加到 target image feature。
4. 两个视角的图像特征分别下采样，再沿宽度拼接，经过 `self_attn_img`，之后上采样并残差回原图像 Token。
5. Token Self-Attention。
6. Token-to-Image Cross-Attention。
7. Token FFN。
8. Image-to-Token Cross-Attention。
9. Transformer 最后还有一次 `final_attn_token_to_image`。

**重要纠正：** `self_attn_img` 的 Self-Attention 序列中同时包含 source 与 target 的压缩图像 Token，因此存在跨视角图像 Token 交互；不能仅凭结构就断言某个方向的注意力一定负责恢复。要用受控 Attention mask 干预验证。

### 1.6 训练时 Refinement 的真实逻辑

**文件：** `vendor/VGGT-S-official/src/main_exp/hybrid.py`  
**函数：** `train_epoch(...)`

当前训练逻辑中：

- `refine_num = int(B * refine_ratio)`。
- 一部分 batch 样本走直接预测分支，`tar_mask_b=None`。
- 另一部分样本在 `torch.no_grad()` 下先执行 `refine_time` 次前向，获得 low-resolution mask。
- 随后使用该 low-resolution mask 再执行最终预测。
- Loss 对最终预测 Mask 计算，GT 是完整目标可见 Mask。

当前配置 `refine_ratio=0.5`、`refine_time=1`。这说明训练中模型确实见过“先预测，再把预测 Mask 作为目标 dense prompt 再预测”的机制。不要把它描述成训练时所有样本都经过 Refinement。

---

## 2. 核心实验原则：把“分块恢复”定义清楚

### 2.1 GT 显著分块定义

必须复用 `scripts/common.py::metrics()` 当前定义：

- 对 target visible GT 使用 8 邻域连通分量。
- 显著面积阈值：`max(10, ceil(0.01 * visible_area))`。
- CR@50 / SCR@50 / `all_components_covered50` 的覆盖阈值为 50%。
- 不修改任何阈值，不另造一套主指标。

建议在 `scripts/vggts_component_analysis.py` 中复用同一逻辑或提取公共 helper，但不得让 GT 参与预测过程。

### 2.2 分块是否有候选点支持

对于每个 GT component `C_k`，统计：

- `candidate_support_count`：40 个目标候选点落在该 GT component 内的数量。
- `selected_prompt_support_count`：最终 5 个目标提示点落在该 component 内的数量。
- `has_candidate_support = candidate_support_count > 0`。
- `has_selected_prompt = selected_prompt_support_count > 0`。

坐标必须使用同一坐标系。建议在目标 crop 画布上比较候选点和 GT crop；如果 GT 目前只保存在原图坐标，应严格复用 `meta` 执行相同 crop transform。不能混用 crop 坐标和原图坐标。

提示点落入 GT component 是事后统计，不是提示点选择的依据。

### 2.3 单个分块的恢复定义

对预测 Mask `P` 与显著 GT component `C_k`：

```text
component_coverage = count(P ∩ C_k) / count(C_k)
component_recovered50 = component_coverage >= 0.5
```

至少报告以下分组：

1. 有候选点支持的分块。
2. 无候选点支持的分块。
3. 有最终 prompt 支持的分块。
4. 无最终 prompt 支持的分块。
5. 最小显著分块（与 SCR@50 一致；并列最小面积 component 都纳入）。
6. 按面积分位数分组的 component。

**无提示点分块恢复率**：

```text
recovery_rate_no_selected_prompt =
    recovered_no_selected_prompt_components /
    all_no_selected_prompt_components
```

同时另报无候选点支持分块恢复率，不能将“无最终 prompt”和“无候选点”混为一类。

### 2.4 样本宏平均与分块微平均都要报告

同一个样本可能有很多 components。为避免 component 多的样本支配结果，至少同时输出：

- component micro average：所有 component 汇总。
- per-sample macro average：先在每个样本内计算，再对样本平均。

样本级“全分块覆盖成功”继续使用现有 `all_components_covered50` 定义。

---

## 3. 实验 A：恢复发生在第一次预测还是 Refinement？

**优先级：P0，第一批必须完成。**  
**不训练，只用现有 checkpoint。**

### 3.1 研究问题

- 第一轮直接 Mask Prediction 是否已经恢复无提示点分块？
- 第二轮使用第一次 Mask 作为 target dense prompt 后，是否新增恢复了分块？
- 新增恢复是否来自真实的 Mask feedback，而不是重复执行一次网络？

### 3.2 重要实现纠正

在当前模型里，`tar_mask_b=None` 时 PromptEncoder 返回 `tar_dense_embeddings_img_b=None`；TwoWayAttentionBlock 第一层只在它不为 None 时把 target mask embedding 加到 target image feature。

因此：

- “第二次仍然传 `None`”是一个 no-feedback 重复前向控制。
- 它不是“无 Mask 反馈但保留同样 target dense embedding 的等价替代”。
- 模型处于 eval、无随机层且输入相同时，重复 None 前向原则上应与第一次输出一致（允许数值误差）。这一组用于检查确定性/重复调用，不应被解释为一个新机制。

### 3.3 实验组

| 组别 | 第一次调用 | 第二次调用 | 目的 |
|---|---|---|---|
| A0: Pass-0 | `tar_mask=None`，保存输出 | 不需要 | 测量初始恢复 |
| A1: Native refinement | `tar_mask=None` | `tar_mask=pass0_low_logits` | 原始完整推理流程 |
| A2: Repeat-no-feedback | `tar_mask=None` | `tar_mask=None` | 控制重复前向/确定性 |
| A3: Corrupted-feedback control | `tar_mask=None` | 输入面积近似匹配但空间位置打乱/平移的 pass0 mask | 检查真实空间反馈是否重要 |

A3 需要明确实现方式：

- 优先对 pass0 low-res logits 做固定、可复现的空间平移或空间置换。
- 保持张量尺寸、dtype、数值范围与原始 low logits 一致。
- 不读取 GT 来决定平移方向、区域或幅度。
- 记录变换参数。
- 若使用跨样本置换，必须固定 permutation，且不要将其他样本 GT 或标签带入。
- 不要将“全零 tar mask”直接等同于 A2；它会进入 mask encoder，和 `None` 的语义不同。

### 3.4 必须保存

每个样本：

- `pass0_logits`、`pass0_low_logits`。
- `pass1_native_logits`、`pass1_native_low_logits`。
- `pass1_repeat_none_logits`。
- `pass1_corrupted_feedback_logits`。
- 对应二值预测 Mask。
- crop recovery 后原图尺寸预测。
- 六项指标（每一 pass 单独计算）。
- 每个 GT component 的 coverage、recovered50、是否有 candidate / selected prompt。

### 3.5 核心分析

对 pass0 失败、pass1 成功的 component，统计新增恢复率：

```text
newly_recovered_components =
    count(pass0_coverage < 0.5 and pass1_coverage >= 0.5)

refinement_rescue_rate =
    newly_recovered_components /
    count(pass0_coverage < 0.5)
```

同时统计退化：

```text
regressed_components =
    count(pass0_coverage >= 0.5 and pass1_coverage < 0.5)
```

对无 selected prompt 的 components 单独计算上述值。

**解释：**

- A1 明显优于 A0，而 A2 与 A0 相同：支持真实 Mask feedback 有贡献。
- A1 与 A3 接近：真实空间反馈的作用可能不强，需进一步检查。
- A1 的无提示 component recovery 提升，但误分割/Boundary F 变差：可能是 Mask 扩张，不等于更准确的分块理解。
- A1 与 A0 几乎一致：Refinement 不是当前恢复现象的主要来源。

---

## 4. 实验 B：哪些输入信息支撑无提示分块恢复？

**优先级：P1。**  
在 A 组确认推理流程正确后执行。

### 4.1 共同控制

- 每组同一个 checkpoint、同一批 val samples、同一 VGGT feature cache。
- 同一组源点、目标点、源点特征。
- 除被消融项外，其他输入完全不变。
- 所有组默认使用原始两次 refinement 流程；同时对最关键组额外记录 Pass-0，以便区分第一次与最终恢复。
- 不改变 sparse token 数量，优先在 embedding 生成后置零对应 token embedding，保持 shape。
- 任何置零干预都必须记录为“zero-ablation”，不要将其解释为完全移除模块的自然分布。
- 同时增加适当的 shuffled / spatially permuted control，以减少零值输入分布外问题。

### 4.2 分组

| 组别 | 干预 | 具体含义 |
|---|---|---|
| B0 Full | 无 | 完整模型基准 |
| B1 No target-point embedding | 将 `tar_points_embed` 置零 | 目标点坐标 Token 的贡献 |
| B2 No source-point coordinate embedding | 将 `src_points_embed` 置零 | 源点空间坐标 Token 的贡献 |
| B3 No source-point feature embedding | 将 `src_points_feat_b` 的 embedding 置零 | 源点局部 VGGT 特征 Token 的贡献 |
| B4 No source dense mask | 将源 Mask 替换为全零 Mask（同时另做 shuffled mask control） | 源 Mask dense prior 的贡献 |
| B5a No source dense image feature | 在进入 MaskDecoder 前只干预 source image feature，保留 `src_points_feat_b` | 源图全局 dense feature 的贡献 |
| B5b No source image information | 同时干预 source dense feature 与 source point feature | 源视角全部视觉信息的贡献 |
| B6 Target-only diagnostic | 去掉 source Mask、源点、目标点、源点特征及 source image feature | 只保留 target image feature 时的模型偏置/先验诊断 |

### 4.3 实现位置

**文件：** `vendor/VGGT-S-official/src/model/prompt_encoder.py`  
**函数：** `PromptEncoder.forward()`

推荐通过独立实验包装器或 forward hook 在 embedding 输出后干预：

- `src_points_embed`
- `tar_points_embed`
- `src_points_feat_b`
- `src_dense_embeddings_img_b`

不要为了实验永久改官方 vendor 源码。若 hook 无法安全访问局部中间变量，可在项目层复制一个可审计的 PromptEncoder 实验 subclass，或增加可选的实验开关，但必须保证默认路径逐位等价。

**文件：** `vendor/VGGT-S-official/src/model/mask_decoder.py`  
**函数：** `MaskDecoder.forward()`

`feat_map_b[:, 0]` 是 source feature，`feat_map_b[:, 1]` 是 target feature。B5a/B5b 可在 MaskDecoder 入口创建干预后的 feature tensor，不能原地修改共享的缓存 Tensor。

### 4.4 必须谨慎解释

- B1 下降：说明目标点 Token 对预测有贡献；不等于模型不能恢复无提示分块。
- B4 下降：支持源 Mask 提供物体身份/形状先验；但全零 Mask 是强分布外干预，必须结合 shuffled mask control。
- B5a 下降：说明 source dense feature 重要；但若 B5b 下降更大，局部 source point feature 也有额外贡献。
- B6 仍输出较大 Mask：不能称为跨视角识别成功；可能只是 target-only foreground bias。必须检查 precision、Visible IoU、component coverage 与 false-positive。
- 单项消融的下降幅度不能直接相加；模块之间存在交互，需要后续组合实验。

---

## 5. 实验 C：Transformer 与 Mask Token 的机制分析

**优先级：P2。**  
只在 B 组发现明确依赖后执行，不要第一轮就全部实现。

### 5.1 C1：source-target image self-attention 是否必要

**文件：** `vendor/VGGT-S-official/src/model/transformer.py`  
**函数：** `TwoWayAttentionBlock.forward()`

当前代码把 source / target image feature 分别下采样，再沿宽度拼接后送入 `self.self_attn_img`。其 attention token 序列同时包含两个视角。

建议实现三组：

- C0：原始 joint source-target image self-attention。
- C1：阻止 source token 与 target token 跨视角互相注意，但保留各自视角内部 self-attention。
- C2：仅允许 target image token 读取 source image token，禁止反方向。
- C3：仅允许 source image token 读取 target image token，禁止反方向。

注意：普通 `Attention.forward()` 当前未暴露 attention mask 参数。若要实现 C1-C3，应新增实验专用 attention mask 路径或独立实验 Attention wrapper；不能简单删除 source feature，也不能把 attention 输出置零后声称是方向性消融。

必须核对 attention logits 的 token 顺序、source/target token 数量和 downsample 后的空间宽度，保证 mask 维度准确。

### 5.2 C4/C5：Token-to-Image 与 Image-to-Token

在 `TwoWayAttentionBlock.forward()` 中：

- `cross_attn_token_to_image`：prompt/token queries 读取 image keys/values。
- `cross_attn_image_to_token`：image queries 读取 prompt/token keys/values。

分别干预：

- C4：关闭 Token-to-Image 输出（保留残差、Norm、FFN）。
- C5：关闭 Image-to-Token 输出（保留残差、Norm、FFN）。

再考虑只对某一层干预与所有层干预的区别。

不能只根据 attention map 的可视化下结论；必须比较真实预测和 component recovery。

### 5.3 C6：Mask Token 输出

**文件：** `vendor/VGGT-S-official/src/model/mask_decoder.py`  
**函数：** `MaskDecoder.forward()`

当前 `num_mask_tokens=1`。Mask Token 经 Transformer 更新后，送入 `output_hypernetworks_mlps` 生成动态权重，再与目标侧 updated feature map 做位置点积。

可做：

- 记录 Mask Token 在每层的 embedding norm / cosine change。
- 记录其对各 GT component 的预测 logit 响应（GT 只在输出后用于统计）。
- 对最终 Mask Token 做 zero / shuffled embedding control（仅作诊断，明确属于分布外干预）。
- 比较 Pass-0 与 Pass-1 的 Mask Token 表征变化。

**不要将 attention weight 直接解释成因果贡献。** C1-C6 的结论必须由干预前后输出差异支持。

---

## 6. 实验 D：按分块属性分析恢复规律

**优先级：P1，可与 A/B 的输出分析并行，不需要额外模型前向。**

### 6.1 候选点支持分组

对每个 GT component 分为：

- D1：40 个候选点支持数 ≥ 2。
- D2：40 个候选点支持数 = 1。
- D3：40 个候选点支持数 = 0。

同时按最终 5 个 target prompt 是否落入该 component 再分层。

不能将 D3 和“没有最终 prompt”合并成一个统计，因为前者更严格。

### 6.2 面积分组

对 component 面积 `area_k` 计算其在样本 visible GT 中的比例：

```text
area_ratio = component_area / visible_mask_area
```

建议按验证集 component 面积分位数（例如 Q1/Q2/Q3）分组；阈值在 val 上确定并记录，不要手工挑选只对某个样本有利的阈值。

### 6.3 分块间距离

对同一样本中 component pair 计算最短像素距离：

```text
d(i,j) = min_{p in C_i, q in C_j} EuclideanDistance(p,q)
```

按 val 分位数分组，分析远距离分块是否更难恢复。

### 6.4 分块数量与形态

记录：

- 每样本显著 component 数量。
- component 面积排序。
- 最小/最大面积比。
- component pair 最短距离分布。
- 每个 component 的候选点支持数。
- 每个 component 的 prompt 支持数。
- Pass-0 / Pass-1 coverage。

### 6.5 遮挡程度代理变量

如果现有数据没有明确 occlusion ratio 标签，不得伪造真实遮挡率。可先使用：

- visible mask area ratio（如 manifest 已有定义）。
- component count。
- component area distribution。
- component pair distance。

这些只能称为遮挡/碎片化的 proxy，不是直接标注的遮挡率。

---

## 7. 六项主指标与新增机制指标

### 7.1 六项主指标：完全复用现有实现

**文件：** `scripts/common.py`  
**函数：** `metrics(pred, gt)`、`summarize(rows)`

保持以下六项不变：

1. `visible_iou`
2. `dice`
3. `boundary_f`
4. `cr50`
5. `scr50`
6. `all_components_covered50`

不要修改 8 邻域连通分量、显著面积阈值 `max(10, ceil(1% visible area))`、Boundary F 容差或 50% coverage 门槛。

### 7.2 新增机制诊断指标（不替代六项主指标）

建议逐 component 输出：

- `candidate_support_count`
- `selected_prompt_support_count`
- `has_candidate_support`
- `has_selected_prompt`
- `pass0_coverage`
- `pass1_coverage`
- `pass0_recovered50`
- `pass1_recovered50`
- `newly_recovered`
- `regressed`
- `component_area`
- `component_area_ratio`
- `nearest_component_distance`

建议汇总：

- `recovery_rate_no_candidate`
- `recovery_rate_no_selected_prompt`
- `recovery_rate_with_candidate`
- `recovery_rate_with_prompt`
- `refinement_rescue_rate`
- `refinement_regression_rate`
- `smallest_component_recovery`
- `component_micro_recovery`
- `component_macro_recovery`

其中：

```text
recovery_rate_no_candidate =
    recovered components with candidate_support_count == 0
    / all components with candidate_support_count == 0
```

```text
recovery_rate_no_selected_prompt =
    recovered components with selected_prompt_support_count == 0
    / all components with selected_prompt_support_count == 0
```

```text
refinement_rescue_rate =
    count(pass0_coverage < 0.5 and pass1_coverage >= 0.5)
    / count(pass0_coverage < 0.5)
```

```text
refinement_regression_rate =
    count(pass0_coverage >= 0.5 and pass1_coverage < 0.5)
    / count(pass0_coverage >= 0.5)
```

分母为 0 时写 `null`，不能写 0 或 NaN。

### 7.3 样本宏平均

每个样本先计算自己的 component recovery rate，再对样本做宏平均。同时报告 component micro average。两者不能混淆。

---

## 8. 实施文件树与具体修改位置

第一轮建议只新增项目层实验代码，不永久修改 vendor 模型结构。

```text
scripts/
├── vggts_official_data.py          # 只读参考；默认不修改
├── common.py                       # 复用 metrics/summarize；不改指标定义
├── vggts_ablation.py               # 新增：A/B 组实验编排
├── vggts_ablation_hooks.py         # 新增：Prompt / feature / refinement 干预
├── vggts_component_analysis.py     # 新增：GT component 事后分析
└── run_vggts_ablation.py           # 新增：CLI 入口

configs/
└── ablations/
    ├── pass_analysis.json          # A0-A3
    ├── prompt_ablation.json        # B0-B6
    └── transformer_ablation.json   # C 组（后续）

tests/
├── test_vggts_ablation.py          # 新增
├── test_vggts_component_analysis.py# 新增
└── test_vggts_ablation_hooks.py    # 新增

outputs/
└── vggts_official/
    └── decoder_mechanism/
        └── <run_id>/
            ├── config.json
            ├── checkpoint_info.json
            ├── sample_manifest.json
            ├── pass_predictions/
            ├── component_records/
            ├── metrics/
            ├── visualizations/
            ├── logs/
            └── diagnostic_report.md
```

### 8.1 `scripts/vggts_ablation.py`

新增函数：

- `build_ablation_case(name, base_inputs, config)`：构造指定干预条件。
- `run_pass_sequence(evaluator, prepared, ablation, save_intermediates)`：执行 Pass-0 / Pass-1，并保存每轮 logits。
- `run_ablation_case(...)`：运行一个实验组。
- `validate_ablation_inputs(...)`：确保样本、checkpoint、features、points 一致。
- `save_prediction_artifacts(...)`：保存 crop logits、原图恢复 Mask 和元数据。

原因：现有 `OfficialEvaluator.predict()` 只返回最终二值 Mask 和 audit，无法单独分析每轮预测。新增独立 runner，避免改变正式评估路径。

### 8.2 `scripts/vggts_ablation_hooks.py`

新增函数：

- `zero_sparse_embedding_group(...)`
- `replace_source_mask(...)`
- `replace_source_feature(...)`
- `make_corrupted_feedback(...)`
- `register_prompt_embedding_hook(...)`
- `register_feature_intervention_hook(...)`
- `register_attention_intervention(...)`（C 组需要时再实现）

原因：把消融逻辑集中管理，避免在官方模型文件中散落永久实验分支。

注意：如果 hook 修改输出，必须返回新 Tensor，不要原地修改缓存或共享 Tensor。所有 hook 要在实验结束时移除。

### 8.3 `scripts/vggts_component_analysis.py`

新增函数：

- `extract_significant_components(gt_visible_mask)`：严格复用 common.py 组件定义。
- `map_points_to_components(points, component_labels, transform)`：点与 GT component 对应。
- `component_coverage(pred_mask, component_mask)`：计算 component coverage。
- `compare_component_passes(pass0, pass1, components)`：新增恢复/退化。
- `summarize_component_groups(records)`：支持组间宏/微平均。
- `render_component_visualization(...)`：GT、候选点、prompt、各 pass 预测分层绘制。

原因：六项指标不能单独回答“哪个分块在第几轮被恢复”，需要逐 component 诊断表。

### 8.4 `scripts/run_vggts_ablation.py`

新增 CLI 参数建议：

```bash
python scripts/run_vggts_ablation.py \
  --mode pass-analysis \
  --split val \
  --checkpoint <formal_checkpoint.pth> \
  --config configs/vggts_official.json \
  --output outputs/vggts_official/decoder_mechanism/<run_id> \
  --limit 300
```

支持模式：

- `pass-analysis`：A 组。
- `prompt-ablation`：B 组。
- `component-analysis`：只对已有预测做 D 组统计。
- `transformer-ablation`：C 组，第二轮再实现。

必须支持 `--limit`，第一轮小规模 smoke 通过后再完整 val。

---

## 9. 关键实验控制与实现风险

### 9.1 正式 checkpoint 与配置校验

- 使用当前正式 VGGTS-S MUVA checkpoint。
- 记录 checkpoint 文件 SHA256。
- 记录 VGGT checkpoint SHA256。
- 记录配置 SHA256、manifest SHA256、代码 commit。
- 不加载 smoke / synthetic checkpoint。
- 不修改正式 checkpoint。
- 不执行 optimizer、backward 或 scheduler。

### 9.2 预处理与缓存一致性

所有消融组必须复用同一个 prepared sample：

- 相同 reference image。
- 相同 reference mask。
- 相同 target crop。
- 相同 target crop metadata。
- 相同 VGGT feature map。
- 相同 source points。
- 相同 target points。
- 相同 source point features。

不要让每个 ablation case 重新采样源点或重新生成 target points。

### 9.3 输入坐标与 Mask 类型

- 保持源/目标点坐标格式、顺序、dtype、device 与原 Predictor 一致。
- `src_mask_b` 当前 Predictor 入口使用 NumPy Mask，再转 Tensor；消融包装器不得误传成不兼容类型。
- `tar_mask_b` 是 low-resolution logits，不是二值 GT Mask。
- 不要把 target visible GT 输入 `tar_mask_b`。
- 不要将最终原图坐标的 GT 与 crop 坐标下的 prompt 混用。

### 9.4 Refinement 的公平比较

- Pass-0 和 Pass-1 必须使用同一 prepared features / points。
- A1 使用真实 Pass-0 low logits。
- A2 使用 None 重复前向，作为 deterministic control。
- A3 使用预先定义、与 GT 无关的 corrupted feedback。
- 保存每次前向的输入摘要，确认没有意外变化。
- `eval()` + `torch.no_grad()`，固定 autocast 设置。
- 记录重复 Pass 的 max absolute logit difference，以验证确定性。

### 9.5 B 组 zero ablation 的分布外风险

将 embedding 置零并不等于自然缺少该信息。必须：
- 记录为 zero ablation。
- 至少对关键输入增加 shuffled/permuted control。
- 不对单个 zero ablation 的下降做过度因果解释。
- 保持 token 数量、维度和位置编码结构一致。

### 9.6 源 Mask 消融的控制

至少做：
- Zero source mask。
- Spatially shifted source mask（保持面积和形状近似，但不对应当前物体）。
- 可选同面积随机形状 mask。

所有变换固定随机种子，不读取 GT 决定变换。

### 9.7 数据泄漏防范

`read_inputs(row)` 只读取 reference RGB、reference Mask、target RGB；`read_gt(row)` 只在预测完成后调用。

不要在 `build_ablation_case()` 或任何模型前向函数中读取 `row['supervision']`。

建议代码层将 GT analysis 与 model inference 分为两个独立阶段：
1. 先完成所有预测并保存。
2. 再加载预测与 GT 进行事后 component analysis。

---

## 10. 输出文件与日志规范

每次运行必须生成：

### 10.1 `config.json`

记录：
- ablation group。
- checkpoint path + SHA256。
- VGGT checkpoint SHA256。
- project commit / source file hashes。
- split / sample IDs / seed。
- refine_time。
- autocast dtype。
- 所有干预项及参数。

### 10.2 `pass_metrics.csv`

逐 sample、逐 pass、逐 ablation 输出六项主指标：

- `visible_iou`
- `dice`
- `boundary_f`
- `cr50`
- `scr50`
- `all_components_covered50`

另保留现有 common.py 的其他诊断字段，但不修改六项定义。

### 10.3 `component_records.csv`

每行一个 sample-component-ablation-pass，至少包含：

```text
sample_id
ablation
pass_index
component_id
component_area
component_area_ratio
candidate_support_count
selected_prompt_support_count
has_candidate_support
has_selected_prompt
coverage
recovered50
is_smallest_component
nearest_other_component_distance
```

### 10.4 `recovery_transition.csv`

每行一个 sample-component，包含：

```text
sample_id
component_id
ablation
pass0_coverage
pass1_coverage
pass0_recovered50
pass1_recovered50
newly_recovered
regressed
has_candidate_support
has_selected_prompt
```

### 10.5 `sample_comparison.csv`

每行一个 sample，包含：
- Pass-0 六项指标。
- Pass-1 六项指标。
- 该样本 no-candidate component 数量。
- no-candidate recovered 数量。
- no-prompt component 数量。
- no-prompt recovered 数量。
- 新恢复数、退化数。

### 10.6 `attention_and_token_diagnostics.csv`（C 组）

可记录：
- layer index。
- token group。
- embedding norm。
- cosine similarity between passes。
- attention output norm。
- source/target token response statistics。

不要保存巨大原始 attention tensor，除非明确有必要；先保存汇总统计和少量样本可视化。

### 10.7 可视化

每个抽样样本保存一张多面板图：

1. Target RGB。
2. GT visible mask + component IDs。
3. 40 candidate points（颜色区分 GT component，仅用于诊断）。
4. 5 selected prompts。
5. Pass-0 prediction。
6. Pass-1 native prediction。
7. Pass-1 corrupted-feedback prediction。
8. Pass-0 / Pass-1 difference map。

候选点颜色可以依据 GT component 事后着色，但必须明确该颜色只用于可视化，不参与选择。

---

## 11. 执行顺序与命令

### Phase 0：代码审计与 Full Model 复现

先不实现 B/C。

- 检查 `OfficialEvaluator.predict()`。
- 检查 `Predictor.forward()`。
- 检查 PromptEncoder / MaskDecoder / Transformer。
- 运行少量 val 样本，确认新增 runner 的 Full Model Pass-1 输出与现有 OfficialEvaluator 输出一致。

验收：
- 同一 checkpoint。
- 同一输入。
- 原有最终预测与新 runner 最终预测逐像素一致；若因 autocast/恢复路径产生数值差异，给出容差和原因。
- 六项指标一致或差异在明确的数值容差内。
- 不更新任何模型参数。

### Phase A：Pass / Refinement 机制

先取约 300 个 val 样本，优先覆盖：
- 多显著分块。
- 至少一个无候选点支持分块。
- 有候选点覆盖所有分块的控制样本。
- 不同 component 数量、面积和距离。

建议命令（由 Agent 根据项目 CLI 实际实现后确认）：

```bash
python scripts/run_vggts_ablation.py \
  --mode pass-analysis \
  --split val \
  --checkpoint <formal_checkpoint.pth> \
  --config configs/vggts_official.json \
  --output outputs/vggts_official/decoder_mechanism/pass_A_<run_id> \
  --limit 300
```

A 组验收：
- A0/A1/A2/A3 每组样本数相同。
- Pass-0 与 Pass-1 logits 均保存。
- A2 重复 None 前向差异接近数值误差。
- A1 与 A0 的 component recovery transition 完整。
- GT 没有进入模型前向。

A 组报告后暂停，先解释结果，再决定 B 组是否全部执行。

### Phase B：输入信息消融

只在 A 组完成并确认 runner 正确后执行。

建议先执行：
- B0 Full。
- B1 No target point embedding。
- B3 No source point feature embedding。
- B4 No source mask + shifted mask control。
- B5a/B5b source feature ablation。

B2 与 B6 可作为补充；B6 只用于目标图像偏置诊断。

### Phase C：Transformer 机制消融

只有 B 组指出明确依赖时再实施。C 组可能需要 attention mask 代码，先在独立 wrapper 中验证 shape 和数值，再进行正式推理。不得在本轮直接改训练代码或重新训练。

### Phase D：分块属性分析

对 A/B 组已有预测离线运行，不需要额外模型前向。

### 最终 test

本轮所有机制选择、阈值和解释均基于 val。待方案冻结后，再按原项目协议一次性运行 test；不要继续用 test 选择实验方向。

---

## 12. 结果解释矩阵

| 观察结果 | 可支持的解释 | 不应直接声称 |
|---|---|---|
| Pass-0 已恢复大量无点分块 | 初始跨视角特征/源 Mask/Token 机制能在第一轮恢复部分分块 | “模型完全不需要点” |
| Pass-1 明显新增恢复，A2 无变化 | 真实预测 Mask feedback 对恢复有贡献 | “Refinement 一定理解了物体结构” |
| B4 源 Mask 消融显著下降 | 源 Mask dense prior 对恢复重要 | “源 Mask 是唯一来源” |
| B5 source feature 消融显著下降 | 源视角特征对恢复重要 | “所有恢复都来自 source-target attention” |
| B1 去掉目标点仍接近 Full | 目标点 Token 对这类恢复不是主要必要条件 | “目标点对整体任务没有用” |
| C1 禁止跨视角 image self-attention 后下降 | 跨视角图像 Token 交互可能重要 | 仅凭 attention map 证明因果 |
| 无点分块恢复率随面积增大而提高 | 模型对大区域更容易恢复 | “小分块无法恢复” |
| 远距离分块恢复率下降 | 空间分散可能是困难因素 | “模型只依赖局部连续性” |
| 所有消融影响都较小 | 恢复可能由多个机制冗余支持，或干预强度不足 | “网络没有使用这些模块” |

### 特别注意：模块交互

如果源 Mask、source feature、refinement 单独消融都产生中等下降，不能简单把下降相加。后续可设计 2×2 factorial experiment，例如：

- Source Mask on/off。
- Refinement feedback on/off。

观察二者是否互补或存在协同效应。

---

## 13. 这项研究如何服务最终课题

最终课题目标不是“让 DBSCAN 产生更多簇”，而是：

> 在多视角遮挡导致同一物体呈现多个空间不连续可见分块时，模型能够发现这些分块、将其关联到同一实例，并完整分割所有可见区域，减少小分块遗漏和遮挡物误分。

本阶段的直接产出是一个机制解释：

- 哪些分块能在没有提示点时恢复？
- 恢复发生在第几次预测？
- 依赖哪些输入信息？
- 依赖哪些网络交互？
- 哪些分块属性决定恢复难度？

这些结论将决定后续新模块应放在哪里：

- 如果源 Mask dense prior 是关键：考虑改进 Mask Prompt Fusion。
- 如果跨视角 image interaction 是关键：考虑多分块身份关联/特征聚合。
- 如果 Refinement 是关键：考虑让 Refinement 显式检查遗漏分块，而不只是细化已有预测。
- 如果小分块始终难恢复：再研究小分块感知的训练目标或分块级辅助监督。

**本阶段不预先指定新模块，也不预先承诺性能提升。** 先完成机制验证，再提出可被实验检验的新方法。

---

## 14. Agent 最终交付清单

必须交付：

1. `diagnostic_report.md`：逐项回答研究问题，区分代码事实、实验结果和假设。
2. `pass_metrics.csv`：A 组逐样本逐 pass 六项指标。
3. `component_records.csv`：逐分块恢复记录。
4. `recovery_transition.csv`：Pass-0 到 Pass-1 的新增恢复与退化。
5. `sample_comparison.csv`：逐样本对照。
6. `config.json`、`checkpoint_info.json`、`sample_manifest.json`。
7. 代表性可视化。
8. 单元测试与 smoke test 日志。
9. 修改文件清单和 Git diff。
10. 明确说明是否改动 vendor 文件、是否改动 checkpoint、是否发生训练（预期均为否）。

**完成 Phase A 后必须暂停并提交报告，不要自动开始 B/C，不要自动训练。**

---

## 15. 给 Agent 的最短执行指令

请按本文件第 1～14 节执行 VGGT-S 无提示点可见分块恢复机制研究。

先审查代码，确认 `OfficialEvaluator.predict()` 的真实两次前向行为，并新增独立实验 runner 保存 Pass-0、Pass-1 与受控 feedback 输出。先完成 Phase A（A0/A1/A2/A3），使用正式 checkpoint 和 val split，不训练、不修改权重、不修改六项指标定义。

GT 只在预测全部保存后用于逐 component 事后分析。必须输出逐样本和逐分块记录、六项主指标、无候选点/无提示点分块恢复率、Refinement 新增恢复与退化率。

Phase A 完成后暂停，交付报告和文件清单，等待用户确认再进入 Phase B。
