# Local data

[中文](README.zh-CN.md)

Third-party images, labels, and training exports stay on disk. Do not commit them. `data/exports/` is ignored on purpose: those JSONL files copy dataset scores and local paths.

```text
data/raw/          official archives or unpacked releases
data/interim/      source catalogs built from those archives
data/processed/    frozen eval items
data/exports/      training exports
data/manifests/    optional extra indices
```

Paths are set in `configs/local.json` (copy `configs/local.example.json`). The image trees themselves live in `datasets/`. See [docs/DATASETS.md](../docs/DATASETS.md).

Code is Apache-2.0. Dataset copyrights stay with their authors.
