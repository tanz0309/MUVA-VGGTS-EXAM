# 第三方代码与许可证

本仓库汇集现有实验适配代码和固定版本第三方源码，不为第三方代码重新授权。各源码目录中的 `LICENSE*`、`NOTICE*`、版权声明及原始归属必须保留；使用、修改和分发遵循各自许可证。

| 快照目录 | 上游 |
|---|---|
| `vendor/INSID3-official` | https://github.com/visinf/INSID3 |
| `vendor/O-MaMa-official` | https://github.com/Maria-SanVil/O-MaMa |
| `vendor/FastSAM-official` | https://github.com/CASIA-IVA-Lab/FastSAM |
| `vendor/VGGT-S-official` | https://github.com/buaa-colalab/VGGT-S |
| `vendor/vggt-official` | https://github.com/facebookresearch/vggt |
| `vendor/V2-SAM-official` | https://github.com/jaychempan/V2-SAM |
| `external/dinov2-runtime` | https://github.com/facebookresearch/dinov2 |
| `external/dinov3-runtime` | https://github.com/facebookresearch/dinov3 |
| `external/mmengine-runtime` | https://github.com/open-mmlab/mmengine |

准确提交、文件摘要、原目录和已有补丁见 `metadata/sources.lock.json` 与 `metadata/source_files.json`。没有 Git 元数据的依赖记录为源码快照，不虚构上游提交。

V²-SAM 官方目录还含 DINO、SAM2、MMDetection/MMEngine 等第三方模块，相关许可与引用继续保留在对应子目录。基础权重不在此仓库，下载和使用需遵循权重提供方条款。

固定版本的 V²-SAM 上游未附根目录许可证；本备份如实保留该状态及其子目录已有许可，不为其补写授权声明。`sources.lock.json` 单独记录实际保留的许可证文件。
