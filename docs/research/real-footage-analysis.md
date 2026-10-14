# Real-footage behaviour of the candidate models

**Status:** evidence produced from committed artifacts (`docs/research/real-footage-analysis.json`)
**Source of numbers:** `scripts/temporal_real_footage_analysis.py`
**Artifacts analysed:** `datasets/external/real-world-eval/{runs,rc1_runs,exp3_runs,exp4_runs,py_runs}/*.json` (git `HEAD`; the underlying videos are not committed)

## What this is (and is not)

Each JSON is a full run of the **C++ ONNX Runtime** over one real external
video, recording per-frame `P(failing)` and the temporal FSM state. This lets us
state how the shipped models behave on real, out-of-domain footage without
re-running inference.

**Two hard limitations, stated up front:**

1. **The video-level labels are VLM-generated, not human-verified.** They come
   from `qwen3-vl-8b` proposals (`docs/research/real-footage-vlm-labels.json`,
   `verification: unverified`). They are a weak reference, not a benchmark. The
   7 source videos are non-redistributable and are not in the repository, so the
   judgements cannot be re-checked from the repo alone.
2. **Only 2 of 7 videos are labelled failing** (87gsL-Luvis, mGH3p3AF1YY), and
   the temporal FSM parameters used by these runs (`use_delta: false`,
   `fail_threshold: 0.3`, `fail_threshold_abs: 0.65`) are **more aggressive than
   the `training/tv/temporal.py` defaults** (`use_delta: true`,
   `fail_threshold: 0.2`, no absolute floor). The video-level alarm counts below
   therefore reflect model scores *and* that aggressive configuration.

## Raw frame scores are the decisive result (config-independent)

Median and tail of `P(failing)` over sampled frames. Failing is the positive
class, so a high value is an alarm. **No temporal layer involved.**

| Video (VLM label) | median baseline | median RC1 | median exp3 | median exp4 | baseline frac ≥0.9 | RC1 frac ≥0.9 |
|---|---|---|---|---|---|---|
| 0TY6LVvMpUo (normal) | 0.983 | 0.962 | 0.000 | 0.171 | 0.76 | 0.58 |
| 38gnsbGqcx4 (normal) | 0.736 | 0.993 | 0.057 | 0.337 | 0.35 | **0.77** |
| G56LQfQCUnA (normal) | 0.933 | 0.883 | 0.377 | 0.367 | 0.54 | 0.48 |
| UzULqvGoi1Y (normal) | 0.772 | 0.881 | 0.087 | 0.535 | 0.31 | 0.46 |
| nbkf372s_cA (normal) | 0.990 | 0.999 | 0.013 | 0.523 | 0.83 | **0.93** |
| 87gsL-Luvis (failing) | 0.995 | 0.990 | **0.000** | 0.156 | 0.64 | 0.79 |
| mGH3p3AF1YY (failing) | 0.975 | 0.788 | 0.577 | 0.679 | 0.81 | 0.50 |

### Findings

1. **The models do not generalise to this footage; they fire "failing" on normal
   video.** Baseline median `P(failing)` is 0.74–0.99 across the five
   VLM-normal videos; on `nbkf372s_cA` 83% of frames score ≥0.9. This is the same
   failure mode as the documented normal-holdout FPR = 1.0, now visible on five
   independent real videos.
2. **RC1 did not fix it, and on two videos it is worse than the baseline it
   replaces.** `38gnsbGqcx4` 0.35 → 0.77 and `nbkf372s_cA` 0.83 → 0.93 (fraction
   of frames ≥0.9). RC1 must not be described as a reliability improvement on the
   basis of the frozen test alone.
3. **exp3 over-corrects into misses.** It suppresses everything — including the
   true failure `87gsL-Luvis`, whose median drops to **0.000** (max 0.45, never
   crossing 0.5). Low false-alarm is achieved by not detecting.
4. **No model cleanly separates failing from normal here.** exp4 is in between
   (normals 0.17–0.54, `87gsL` 0.156). On this evidence there is no
   configuration in the repository that both suppresses normal footage and
   detects the real failures.

## Temporal FSM behaviour (video-level alarm)

`first_failing_event_frame ≥ 0` ⇒ the FSM emitted a FAILING event:

| Model | normal videos alarmed | failing videos alarmed |
|---|---|---|
| baseline | 5/5 | 2/2 |
| RC1 | 5/5 | 2/2 |
| exp3 | 5/5 | 1/2 (misses 87gsL) |
| exp4 | 5/5 | 2/2 |

Under the configuration these runs used, every model alarms on every normal
video (video-level false-alarm rate 1.0), driven by the raw scores above.

## How to reproduce

```bash
training/.venv/bin/python scripts/temporal_real_footage_analysis.py \
    --output docs/research/real-footage-analysis.json
```

The script reads the run JSONs from the working tree when present, otherwise
from `git show HEAD:` (they are deliberately pruned from this worktree because
they derive from non-redistributable footage).

## Consequence for the release decision

This is **real-data evidence** (actual ONNX Runtime runs on real footage) that is
stronger than a synthetic benchmark, but it rests on **unverified labels** and a
**small, single-configuration** set. It is sufficient to *block* the claim
"TinyVision reliably distinguishes normal from failing prints on real footage",
and to *block* the claim "RC1 improved real-domain reliability". It is **not**
sufficient to quantify a precise FPR/FNR, because the labels are not verified.
The release gate must therefore remain INCONCLUSIVE/FAIL on real-domain
reliability, not PASS.
