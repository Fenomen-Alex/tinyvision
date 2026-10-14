# Planned experiment configs (NOT RUN)

These configs are **proposals** for the experiments described in
[`../../../docs/research/model-improvement-plan.md`](../../../docs/research/model-improvement-plan.md).
They are deliberately placed here, not in `training/configs/`, so no existing
run or tool picks them up by glob. Nothing here has been trained or evaluated.

To run one, copy it into `training/configs/` (or pass it explicitly to
`tv train --config`) after its dataset exists and the environment-independent
validation surface is available.
