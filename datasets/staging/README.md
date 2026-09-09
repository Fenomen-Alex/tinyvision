# Dataset staging area

Environment-centric staging for new camera data. See
[`docs/research/dataset-staging-pipeline.md`](../../docs/research/dataset-staging-pipeline.md).

Layout:

```
<dataset>/
    environments/<env_id>.json          derived views of the canonical manifest
    recordings/<rec_id>.json            derived views of the canonical manifest
    manifests/env_manifest.json         canonical manifest (schema tinyvision-env-manifest)
    metadata/                           provenance, inventory, characterization records
    review/                             review-state snapshots
    samples/<recording_id>.jsonl        one frame sample per line
```

* `env_manifest.example.json` — manifest template.
* `readiness/canonical_holdings.json` — already-owned frozen assets used by
  `tv env-readiness`.
* `bambu/` — staged Bambu Timelapse Dataset (CC BY 4.0) environment inventory
  + manifest. Rebuild with `uv run --project training python
  scripts/build_bambu_staging.py`.
* `trier/` — Trier FFF-QC v2 (CC BY 4.0) characterization from the Zenodo API
  record; no media downloaded (48 GB), environment units UNVERIFIED until
  `tv env-index --kind trier --root <extracted>` is run.
* `contrib/` — reserved for contributor datasets.

Workflow: build/refresh metadata with `tv env-index`, write or edit the
canonical manifest (see the example), `tv env-validate`, `tv env-split`, then
track the gate with `tv env-readiness`.