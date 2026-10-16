# datasets/

[English](README.md)

ImageJev 的本地技术 / 感知图像质量文件。这里只有本 README 会被 Git 跟踪。

每个来源保持官方解压后的目录。这些目录里的文件都不提交。这里不划 ImageJev 自己的训练/验证/测试。CID2013 的图像集目录和 UHD-IQA 原来的 training/validation/test 目录只当来源记录。

```text
spaq/Annotations/MOS and Image attribute scores.xlsx
spaq/SPAQ zip/SPAQ/TestImage/*.jpg
koniq10k/koniq10k_1024x768.zip
koniq10k/koniq10k_scores_and_distributions.zip
cid2013/                     解压后的 CID2013.7z，含 MOS 工作簿
uhd-iqa/                     官方 training、validation、test 目录
```

`scripts/build_source_catalog.py` 也接受 `spaq/images` 加 `spaq/annotations`，以及已经解出来、和分数 CSV 放在一起的 KonIQ JPEG。

| id | 角色 |
| --- | --- |
| `spaq` | 手机照片 MOS 和五个属性 |
| `koniq10k` | 野外照片 MOS |
| `koniq++` | 按 KonIQ 文件名连接的缺陷类型比例 |
| `cid2013` | 相机和手机 MOS |
| `uhd-iqa` | 4K MOS |

下载位置和上游条款见 [docs/zh/DATASETS.md](../docs/zh/DATASETS.md)。这些文件的版权属于原作者。
