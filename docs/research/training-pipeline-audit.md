# Training pipeline audit

Scope: `training/tv/{config,train,experiment,export,evaluate,data,models}.py`,
the `tv` CLI, and the experiment configs. This is a read-only audit; no training
was run.

## What the pipeline does

* **Config** (`config.py`): dataclass schema (`DatasetConfig`, `DataConfig`,
  `AugmentationConfig`, `TrainConfig`, `EvalConfig`, `ExportConfig`,
  `HardwareConfig`, `SoftwareConfig`) loaded from YAML. A content-hashed
  `experiment_id` (`_experiment_identity`) covers every result-determining knob
  (model, task, dataset, data, train incl. augmentation, eval threshold, export)
  and deliberately excludes hardware/software/timestamp, so the same config
  yields the same ID on any machine and `evaluate`/`export` can locate the
  checkpoint `train` produced.
* **Train** (`train.py`): deterministic seeding
  (`random`/`numpy`/`torch`/cuda) + `cudnn.deterministic`, a picklable
  per-worker `worker_init_fn`, Adam, early stopping. `precision` field supports
  fp32 (rc1 used fp32).
* **Experiment** (`experiment.py`): `ExperimentManager` writes status, metrics
  and checkpoints under `experiments/<experiment_id>/`.
* **Evaluate / export**: metrics (`precision, recall, f1, fnr, fpr, auroc`) and
  ONNX + INT8 export (`export.quantization`).

Reproducibility posture: **good**. Seeds are set, workers are seeded, IDs are
content-derived, hardware/software are recorded in the config at runtime.

## Experiment matrix (configs present)

| Config | Model | Dataset | Purpose | Outcome |
|---|---|---|---|---|
| `experiment1*.yaml` (b0, m3l, m3s, r18, shufflenet) | 5 backbones | printing-errors | model selection | sh/ m3s carried forward |
| `classification_base.yaml` | default | printing-errors | template | — |
| `exp2_external_video_aug.yaml` | shufflenet | printing-errors + external video aug | external-normal robustness | not released |
| `exp3_target_normal_aug.yaml` | shufflenet | + stage13 normals | target-normal augmentation | not released (tails unsatisfiable) |
| `exp4_target_balanced.yaml` | shufflenet | + stage13/14 balanced | balanced augmentation | not released |
| `skz_expA_external.yaml` / `skz_expB_aux.yaml` | shufflenet | SKZ probe / aux | external generalization probe | probe only |
| `eval_target_camera_shufflenet_v2_x1_0.yaml` | shufflenet | target camera | camera eval | — |
| `robustness_shufflenet_v2_x1_0.yaml` | shufflenet | robustness | perturbation study | — |
| `rc1_release_candidate.yaml` | shufflenet | release-rc1 (frozen + Trier source_annotation + 44 human-verified failing) | first release candidate | **evaluated, NOT released** |

## Findings / risks

1. **Trains and tests on the same venue.** rc1's `val.csv`/`test.csv` are
   byte-identical copies of the frozen printing-errors splits, and rc1 trains on
   printing-errors plus Trier. So the within-suite numbers are single-venue
   (see [`evaluation-independence.md`](evaluation-independence.md)) and cannot
   estimate generalization. This is the mechanism behind the transfer failure.
2. **Model selection on a non-independent val.** Early stopping and threshold
   selection use `dev_validation`, which is same-venue. Acceptable for
   within-venue fit, unsafe as a generalization proxy.
3. **No explicit class weighting / sampler.** With failing 1417 / normal 757 in
   val, the small-model bias toward failing recall at the expense of normal
   precision is expected; the config exposes no `class_weight`/`sampler` knob.
4. **Augmentation is mild** (`horizontal_flip`, `random_rotation=15`,
   `color_jitter=0.2`) and identical across rc1/exp3/exp4, so it does not
   address camera/lighting domain shift — the actual failure mode.
5. **`precision` field is present but fp32-only in practice**; no AMP path
   tested for the edge target.
6. **INT8 export changes FPR (+4 points on dev_val)** — quantization is
   performed but not treated as a distinct model in the training config.

## Recommendations (see model-improvement-plan.md)

* Add domain-randomized augmentation and camera/lighting augmentation.
* Add an environment-independent validation surface before tuning anything.
* Expose class weighting / balanced sampling.
* Treat INT8 as a separately-gated artifact.
* Keep configs reproducible; any augmentation change must be a new config, not
  an edit of an existing run's config.
