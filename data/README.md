# Local data

Third-party images and labels stay on disk. Do not commit them.

```text
data/raw/          immutable official archives / unpacked releases
data/interim/      source catalogs built from those archives
data/processed/    frozen Week 0 eval items and copied assets
data/exports/      later training exports (not used in Week 0)
data/manifests/    optional extra indices
```

Configure paths in `configs/local.json` (copy `configs/local.example.json`).
Raw IQA trees live in `datasets/` (see `datasets/README.md`).
See `docs/DATASETS.md` for official download locations and license caveats.

Code license (Apache-2.0) is not a dataset license. Research-only sources
remain research-only.
