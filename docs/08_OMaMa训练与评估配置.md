# O-MaMa：官方匹配网络与 MUVA 训练

更新于2026-11-17。[作者仓库](https://github.com/Maria-SanVil/O-MaMa)固定提交 `0f187c65cb9d8f8df1d8b5f445edbb0485956d88`，本地 `vendor/O-MaMa-official`，tracked源码不修改。环境 `/opt/conda/envs/omama`。

## 输入与推理

仅输入参考RGB、参考可见掩码、目标RGB。FastSAM根据目标RGB在线生成候选；作者DINOv2描述符及Attention_projector计算匹配，选择一个候选作为预测，再恢复到原图评价。不使用目标GT选择候选，不增加分块合并模块。

保留 reverse=True 空间角色：参考532×952、目标700×700；这是适配原网络固定位置网格的处理，不表示MUVA具备ego/exo语义。图像使用nearest缩放后按作者均值/方差归一化，掩码nearest；FastSAM输入1024、conf0.4、IoU0.9；DINOv2 ViT-B/14带register。骨干冻结，权重复用本地文件。

描述符函数原生使用 `GT_img` 这个键，实际值是**目标RGB**，不包含目标掩码。只替换Torch Hub加载位置为已下载的本地源码，模型及descriptor计算仍调用作者实现。

推理计时包含在线FastSAM、描述符与匹配。没有候选时输出空预测，不用GT补救；审计记录候选数、所选索引及匹配分数。

## 训练

- Attention_projector随机初始化，不加载Exo2Ego.pt或历史任务权重。
- 3轮，单GPU batch24，与原main.py默认值一致。冻结描述符按图对串行提取后沿batch维拼接，未改动的官方匹配头执行一次完整batch前向，使用作者原生均值损失。最后不足24对的batch正常参与训练。
- 每对32候选，训练GT作为强正例插入随机位置，31个负例由作者 `select_adjacent_negatives` 选择。
- 原生候选质心Delaunay及二阶邻接，优先邻近负例，不足则按作者方法补采。共线/重复质心引发Qhull错误时使用零邻接回退，让原选择器从其余候选补采，并记录 `adjacency_fallback`。
- AdamW lr8e-5、默认weight_decay0.01，cosine至1e-6，**不加梯度裁剪**，与官方main.py的更新规则一致。
- 描述符、模型、训练正负例函数、损失均复用作者代码。MUVA迭代与统一评估由本地驱动承接，不声称原main.py只改一个路径即可直接运行。
- 每轮共同val，按Visible mIoU保存best；test不参与选模。正式推理不插入GT正例。

该方法受候选质量限制：若没有一个候选包含全部可见分块，当前基线不额外组合多个候选。这样才能评价作者方法在分块数据上的表现。

## 使用

完整命令见文档07，新服务器GPU检查进度及历史3090证据见文档06。核心命令为 `run.sh train omama`，完成后用本轮 `best.pth` 评价val/test。支持用 `last.pth` 从完整epoch恢复；新协议不接受旧配置或smoke权重。
