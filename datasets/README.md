# datasets/

Local **technical / perceptual IQA** sources for Image-JEV. Not committed.

Layout per source: `raw/` (official archive), `images/`, `labels/` (or `annotations/`).
**No train/validation/test split** is applied here. Official CID2013 image-set folders and UHD-IQA’s original training/validation/test directories are kept only as source provenance.

## Sources in this tree

| id | role | how it got here |
|---|---|---|
| `spaq` | smartphone photo MOS + 5 attributes | Junction of images from `alpha-half-vlm-demo` (same NTFS volume, no extra 33 GiB). Annotation xlsx copied. |
| `koniq10k` | in-the-wild MOS | ZIP hardlinked from `alpha-half-vlm-demo`; JPEGs extracted; score/indicator CSV extracted. |
| `koniq++` | defect-type rates on KonIQ filenames | Copied from `F:\datasets\koniq++`. Join to KonIQ images on `filename`. |
| `cid2013` | authentic camera/phone MOS | Downloaded from Zenodo and unpacked. |
| `uhd-iqa` | 4K authentic MOS (CC0) | Copied last from `F:\datasets\uhd-iqa` **only if D: has enough free space**. |

Composition / aesthetics raw dumps in AlphaHalf (PICD, CADB, MSC, ArtiMuse) are **not** copied; they are not technical-quality gold.

## Inventory

See `inventory.json` (counts, bytes, join stats) and `catalog.jsonl` (one row per labelled image, `split` omitted).

## Disk note

SPAQ images are a **directory junction**, KonIQ official ZIPs are **hardlinks**. Deleting the AlphaHalf originals while these links remain is unsafe for SPAQ (junction target). Hardlinked ZIPs stay valid until the last link is removed.
