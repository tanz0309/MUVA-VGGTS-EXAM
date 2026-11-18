# 环境、源码版本与初始化

## 环境与入口

| 方法 | 入口 | Conda环境 |
|---|---|---|
| INSID3 | `insid3` | `/opt/conda/envs/insid3-muva` |
| O-MaMa | `omama` | `/opt/conda/envs/omama` |
| VGGT-S | `vggts`，同义名`vggts-official` | `/opt/conda/envs/vggts` |
| V²-SAM | `v2sam`、`v2sam-visual`、`v2sam-anchor`、`v2sam-multi` | `/personal_workspace/TanZhuzhu/.conda/envs/v2sam` |

同一篇V²-SAM论文的几个专家共享其专用环境。ObjectRelator暂缓，原代码和环境保留。

`run.sh`直接调用对应环境的绝对Python路径，并检查sys.prefix，无需提前activate。CUDA不可用则停止，不回退CPU模型运算。读图、RLE、聚类、标量距离和指标在CPU处理属于数据工作，不等于CPU神经网络推理。

INSID3为torch2.7.1+cu126，O-MaMa为2.4.0+cu121，VGGT-S及V²-SAM为2.3.1+cu121。`GPU_ID=0`选择物理0号卡；请顺序运行，避免同卡同时挤占显存。新服务器为RTX4090，驱动报告49,140 MiB显存；旧3090检查属于历史证据，新机检查进度见文档06。

## 固定作者源码

以下目录均位于 `/root/datasets/MUVA_experiments/vendor`。启动时校验主模型repo_commit和tracked文件状态，不把旧的个人实验仓库当作官方源码。

| 目录 | 固定提交 |
|---|---|
| INSID3-official | `0c165a10cf52ab91f335883d06260de86854adbe` |
| O-MaMa-official | `0f187c65cb9d8f8df1d8b5f445edbb0485956d88` |
| FastSAM-official | `b4ed20c2fed75eadc5aa7d8b09fedd137b873b52` |
| VGGT-S-official | `7ba82679d662d42171b95f1977f01328f1b18630` |
| vggt-official | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| V2-SAM-official | `50e7c0584e852e1e16703c07e33c49e1d1bde635` |

VGGT的track head按VGGT-S作者README增加特征返回接口，是明确记录的例外。V²-SAM的 `projects/v2sam` 是指向Fusion的包兼容别名，Visual显式调用自身类；tracked源文件不修改。环境中已安装的作者MMEngine来自同版本 `/root/V2-SAM/mmengine`，已核对tracked内容未修改。

旧仓库及已有大权重保留原路径并复用；新运行的配置记录绝对路径及权重SHA256。

## 重新训练的含义

INSID3和Anchor免训练。O-MaMa随机初始化Attention_projector；VGGT-S随机初始化Predictor。V²-SAM新增VPMatcher和视觉投影随机初始化，SAM2解码器从公开SAM2基础权重初始化后训练，编码器冻结。这遵循作者设计，不是把所有骨干从随机参数重新预训练。

所有可训练实验新建优化器，在同一MUVA train上开始，不加载作者已训练好的跨视角任务checkpoint，不从smoke权重继续。正式恢复只接受同模型、同配置、同清单的正式checkpoint。

## 种子及记录

四种方法所有入口统一全局seed42，覆盖Python、NumPy、Torch、CUDA及数据打乱；每轮顺序使用42+从0开始的epoch索引，即首轮42、第二轮43。VGGT-S训练循环从1计数，适配层对应使用42+epoch-1，避免首轮错用43。VGGT-S作者函数内部固定的KMeans seed0保留，不能为“统一种子”擅自改原算法。关闭cuDNN benchmark，开启确定性设置并记录警告；Flash Attention、cumsum和grid_sample等仍可能非逐位确定，不能承诺跨环境bitwise一致。

每次记录有效配置、源码版本、清单与权重摘要、环境、设备和适配代码摘要。PCCS额外记录选择器源码摘要与逐案例评分。只依据当前smoke证明可运行，完整训练效果尚待用户启动实验。
