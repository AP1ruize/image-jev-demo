# datasets/

[中文](README.zh-CN.md)

Local technical / perceptual image-quality files for ImageJev. This README is the only file here that Git tracks.

Keep each source in its official unpacked layout. Nothing under those folders is committed. ImageJev does not assign its own train/validation/test split in this tree. Official CID2013 image-set folders and UHD-IQA’s original training/validation/test directories are source provenance only.

```text
spaq/Annotations/MOS and Image attribute scores.xlsx
spaq/SPAQ zip/SPAQ/TestImage/*.jpg
koniq10k/koniq10k_1024x768.zip
koniq10k/koniq10k_scores_and_distributions.zip
cid2013/                     CID2013.7z unpacked, with the MOS workbook
uhd-iqa/                     official training, validation, and test folders
```

`scripts/build_source_catalog.py` also accepts `spaq/images` plus `spaq/annotations`, and KonIQ JPEGs already extracted beside the score CSV.

| id | role |
| --- | --- |
| `spaq` | smartphone photo MOS and five attributes |
| `koniq10k` | in-the-wild MOS |
| `koniq++` | defect-type rates joined to KonIQ filenames |
| `cid2013` | camera and phone MOS |
| `uhd-iqa` | 4K MOS |

Download locations and upstream terms: [docs/DATASETS.md](../docs/DATASETS.md). Copyright in these files stays with their authors.
