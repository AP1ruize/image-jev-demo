# 数据集

[English](../DATASETS.md)

Git 只跟踪这篇说明，不跟踪文件。`datasets/` 和 `data/` 除了 README 和空占位，都不入库：压缩包、图像、表格、CSV 标注、训练导出都留在本地。

下面是官方解压后的目录。按这些名字放在磁盘上，或者放进 `scripts/build_source_catalog.py` 已经接受的本地目录。两种都不要提交。

Week 0 评的是 **SPAQ + KonIQ-10k** 的总体 MOS。CID2013、KonIQ++、UHD-IQA 先放着，供以后的技术质量工作使用。它们不是审美或构图数据集。

## SPAQ

- 论文：Fang 等，CVPR 2020，[Perceptual Quality Assessment of Smartphone Photography](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)
- 仓库：https://github.com/h4nwei/SPAQ
- 11,125 张手机照片，MOS 以及 Brightness / Colorfulness / Contrast / Noisiness / Sharpness（0–100）
- 从官方 Google Drive 或百度包下载。解压后的预期布局：

```text
<spaq_root>/
  Annotations/MOS and Image attribute scores.xlsx
  SPAQ zip/SPAQ/TestImage/*.jpg
```

有的镜像把 `TestImage` 和 `Annotations` 放高一层；`scripts/build_source_catalog.py` 两种都接受。也可以把 `spaq_manifest` 指到已经含有 `source_id`、分数和图像路径的 JSONL。

数据包 Readme 写的是教育和研究使用，商业使用需要作者许可。这是上游自己的说明。

## KonIQ-10k

- 论文：Hosu 等，https://arxiv.org/abs/1910.06180
- 门户：https://database.mmsp-kn.de/
- 直接文件：
  - https://datasets.vqa.mmsp-kn.de/archives/koniq10k_1024x768.zip
  - https://datasets.vqa.mmsp-kn.de/archives/koniq10k_scores_and_distributions.zip
- 用分数 CSV 的 `image_name` 连接图像 ZIP 里的 JPEG。丢掉 ZIP 里没有 MOS 的 300 张。

## CID2013

- 论文：Virtanen 等，IEEE TIP 2015
- Zenodo：https://zenodo.org/records/2647033
- 直接文件：https://zenodo.org/records/2647033/files/CID2013.7z?download=1
- 474 张相机/手机 JPEG，六个图像集；MOS 工作簿 `CID2013 data - version 12112014.xlsx`

## KonIQ++

- 只有标注：https://datasets.vqa.mmsp-kn.de/archives/koniq++/koniq++database_distributions.csv
- 用 `filename` 连到 KonIQ-10k 的 JPEG（`artifacts`、`blur`、`contrast`、`colors`、`other`）

## UHD-IQA

- 门户：https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html
- 6,073 张 4K 照片，图像为 CC0。解压树放在 `datasets/uhd-iqa/`（约 10 GiB）。官方的 training/validation/test 目录只当来源记录；ImageJev 还没有冻结自己的切分。

## 本地配置

把 `configs/local.example.json` 抄成 `configs/local.json`（不入库）：

```json
{
  "sources": {
    "spaq_root": "datasets/spaq",
    "koniq_images_zip": "datasets/koniq10k/raw/koniq10k_1024x768.zip",
    "koniq_scores_zip": "datasets/koniq10k/raw/koniq10k_scores_and_distributions.zip"
  }
}
```

各来源放在 `datasets/spaq`、`datasets/koniq10k`、`datasets/cid2013`，可选 `datasets/uhd-iqa`。KonIQ++ 标注是 `datasets/koniq10k/labels/koniq++database_distributions.csv`。

如果已经有 SPAQ 的 JSONL：

```json
{
  "sources": {
    "spaq_manifest": "path/to/spaq-source.jsonl",
    "spaq_path_root": "path/to/resolve/relative/image/paths",
    "koniq_images_zip": "...",
    "koniq_scores_zip": "..."
  }
}
```

JSONL 里的相对图像路径，有 `spaq_path_root` 时相对它解析，否则相对 JSONL 所在目录。

## 权利

本仓库代码是 Apache-2.0。数据集的版权和条款属于原作者，本仓库不授予这些权利。下载或训练之前读各上游说明。SPAQ 数据包 Readme 是其中一份：它写教育和研究使用，并写明商业使用需要作者许可。KonIQ、CID2013、UHD-IQA 各有自己的条款。

冻结的测试 `item_id` 不会用于训练。不要把第三方图像提交进本仓库。
