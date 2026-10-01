# Datasets

Prepared local tree: `datasets/` (gitignored except `datasets/README.md`). Inventory: `datasets/inventory.json`. Item list without train/test assignment: `datasets/catalog.jsonl`.

Week 0 still scores **SPAQ + KonIQ-10k** overall MOS. CID2013, KonIQ++, and UHD-IQA are staged for later technical-quality work. They are not aesthetic or composition datasets.

## SPAQ

- Paper: Fang et al., CVPR 2020, [Perceptual Quality Assessment of Smartphone Photography](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)
- Repo: https://github.com/h4nwei/SPAQ
- 11,125 phone photos, MOS plus Brightness / Colorfulness / Contrast / Noisiness / Sharpness (0–100)
- Download the official Google Drive or Baidu package. Expected layout after unpack:

```text
<spaq_root>/
  Annotations/MOS and Image attribute scores.xlsx
  SPAQ zip/SPAQ/TestImage/*.jpg
```

Some mirrors put `TestImage` and `Annotations` one folder higher; `scripts/build_source_catalog.py` accepts either. You can also point `spaq_manifest` at a JSONL that already lists `source_id`, scores, and image paths.

Package Readme: educational/research use; commercial use needs permission.

## KonIQ-10k

- Paper: Hosu et al., https://arxiv.org/abs/1910.06180
- Portal: https://database.mmsp-kn.de/
- Direct files:
  - https://datasets.vqa.mmsp-kn.de/archives/koniq10k_1024x768.zip
  - https://datasets.vqa.mmsp-kn.de/archives/koniq10k_scores_and_distributions.zip
- Join `image_name` in the score CSV to JPEG members of the image ZIP. Drop ZIP-only files with no MOS (300 extras).

## CID2013

- Paper: Virtanen et al., IEEE TIP 2015
- Zenodo: https://zenodo.org/records/2647033
- Direct: https://zenodo.org/records/2647033/files/CID2013.7z?download=1
- 474 camera/phone JPEGs in six image sets; MOS workbook `CID2013 data - version 12112014.xlsx`

## KonIQ++

- Labels only: https://datasets.vqa.mmsp-kn.de/archives/koniq++/koniq++database_distributions.csv
- Join `filename` to KonIQ-10k JPEGs (`artifacts`, `blur`, `contrast`, `colors`, `other`)

## UHD-IQA

- Portal: https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html
- 6,073 4K photos, images CC0. Unpacked tree is on `datasets/uhd-iqa/` (~10 GiB). Official training/validation/test folders are source provenance only; Image-JEV has not frozen a split.

## Local config

Copy `configs/local.example.json` to `configs/local.json` (gitignored):

```json
{
  "sources": {
    "spaq_root": "D:/image-jev-demo/datasets/spaq",
    "koniq_images_zip": "D:/image-jev-demo/datasets/koniq10k/raw/koniq10k_1024x768.zip",
    "koniq_scores_zip": "D:/image-jev-demo/datasets/koniq10k/raw/koniq10k_scores_and_distributions.zip"
  }
}
```

Organized copies (and junctions/hardlinks) live under `datasets/spaq`, `datasets/koniq10k`, `datasets/cid2013`, and optionally `datasets/uhd-iqa`. KonIQ++ labels are `datasets/koniq10k/labels/koniq++database_distributions.csv`.

If you already have a SPAQ source JSONL:

```json
{
  "sources": {
    "spaq_manifest": "D:/path/to/spaq-source.jsonl",
    "spaq_path_root": "D:/path/to/resolve/relative/image/paths",
    "koniq_images_zip": "...",
    "koniq_scores_zip": "..."
  }
}
```

Relative image paths in that JSONL are resolved against `spaq_path_root` when set, otherwise against the JSONL's parent directory.

## Rules

- Code license ≠ data license.
- Frozen test `item_id`s are never used for training.
- Do not redistribute third-party images from this repository.
