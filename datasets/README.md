# `datasets/`

Data is never committed by default; only manifests, labels and provenance are.

* **Policy, canonical manifest schema and tooling**: [`../DATASETS.md`](../DATASETS.md)
* **Current inventory & usability verdicts**: [`../docs/research/dataset-inventory.md`](../docs/research/dataset-inventory.md)
* **Staging intake area**: [`staging/README.md`](staging/README.md)

Layout:

```
raw/            original downloads (not committed)
Printing_Errors/ NilsHagenBeyer "3d printing errors" (Apache 2.0)
processed/      derived, environment-split frames + split CSVs
external/       external holdout + label manifests (not redistributable)
staging/        environment-centric intake (manifests, samples, review)
```

Do not use a dataset whose licence is unverified. Do not split at frame level.
