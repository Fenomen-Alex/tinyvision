# Experiments

Every experiment must record: experiment ID, model + version, dataset + version
and split, image size, epochs, batch size, optimizer, learning rate,
augmentation, hardware, software versions, seed, precision, metrics, model
size, and inference latency. Reproducibility matters more than novelty.

The authoritative history is [`docs/research/rc1-release-candidate.md`](docs/research/rc1-release-candidate.md)
and the neighbouring stage/experiment reports. A pipeline audit and the
per-config matrix are in
[`docs/research/training-pipeline-audit.md`](docs/research/training-pipeline-audit.md).

## How experiments are identified

`training/tv/config.py` derives a deterministic `experiment_id` by hashing every
result-determining knob (model, task, dataset, data, train incl. augmentation,
eval threshold, export). Hardware/software/timestamp are excluded, so the same
config yields the same ID anywhere and `evaluate`/`export` can locate the
checkpoint `train` produced. **Never edit a config after a run** — make a new
config so IDs and results stay unique.

## Running one

```bash
# from the repo root, with the training package installed
python -m tv.cli train    --config training/configs/<name>.yaml
python -m tv.cli evaluate --config training/configs/<name>.yaml
python -m tv.cli export   --config training/configs/<name>.yaml
```

Artifacts land under `experiments/<experiment_id>/` (git-ignored).

## Current configs

| Config | Model | Purpose |
|---|---|---|
| `experiment1*.yaml` | b0 / m3l / m3s / resnet18 / shufflenet | model selection |
| `exp2_external_video_aug.yaml` | shufflenet | external-normal video augmentation |
| `exp3_target_normal_aug.yaml` | shufflenet | target-normal augmentation |
| `exp4_target_balanced.yaml` | shufflenet | balanced augmentation |
| `skz_expA_external.yaml` / `skz_expB_aux.yaml` | shufflenet | SKZ probe / auxiliary |
| `robustness_shufflenet_v2_x1_0.yaml` | shufflenet | robustness perturbations |
| `rc1_release_candidate.yaml` | shufflenet | first release candidate (**not released**) |

Planned, **not run** configs (proposals) live in
[`training/configs/planned/`](training/configs/planned/) and are described in
[`docs/research/model-improvement-plan.md`](docs/research/model-improvement-plan.md).

## Rules

* Evaluation data is separate from training data; split at environment level.
* Do not change model architecture, dataset definition, evaluation criteria or
  benchmark methodology silently.
* Do not report within-venue numbers as generalization (see
  [`docs/research/evaluation-independence.md`](docs/research/evaluation-independence.md)).
* A config that differs in any knob must be a new config, not an edit.
