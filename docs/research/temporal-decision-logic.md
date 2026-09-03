# Temporal Smoothing and Decision Logic for Video-Based Monitoring

**Status:** Research (not implemented)
**Applies to:** 3D-print failure detection (and by extension other video monitoring tasks)
**Related open question:** `problem-selection.md` § "Must resolve during implementation" item 7 (temporal smoothing)

## Context

The monitoring task is: a webcam captures frames at 1–15 FPS of a 3D printer, a per-frame
model produces a confidence score that a failure is occurring, and a decision layer decides
when to trigger an action (notification and/or an automatic print pause).

Key characteristics that shape the decision logic:

- **Failures develop over seconds-to-minutes**, not frames. A few frames of delay in
  triggering is almost always harmless; a spurious trigger (unnecessary pause) wastes
  filament and time but is recoverable.
- **Single-frame predictions are noisy.** Lighting flicker, webcam compression, motion
  blur, and ambiguous geometries all cause occasional high-confidence false positives and
  low-confidence false negatives.
- **Asymmetry of cost.** False positives ("unnecessary pauses") are annoying but
  tolerable. False negatives (missed failures) are costly — wasted filament plus print time,
  potentially an unattended overnight print.

The practical question is **not** whether to smooth — every deployed system does — but
**which** smoothing/decision scheme to use and how to tune it. This document compares the
candidate approaches and reviews what is known from deployment and literature.

---

## 1. The five candidate approaches

All five operate on a per-frame model output. The first four are the same class of
solution (statistical smoothing over the confidence score / binary prediction); the fifth
(FSM) is a structural framework that typically *wraps* one of the others. None are mutually
exclusive — real systems combine them. The realistic design space is: pick a smoothing
method (moving average / exponential smoothing / consecutive-frame), then layer hysteresis
and/or a state machine on top.

### 1.1 Moving average (rolling window)

**How it works.** Maintain a sliding window of the last N frame scores (either the raw
confidence or a binarized 0/1 prediction) and use their mean as the "current" score.
Trigger when the window mean crosses a threshold.

- Score version: `S_t = (1/N) * Σ_{i=t-N+1..t} p_i` — smooths noisy confidence.
- Binarized (majority voting) version: trigger when ≥ K of the last N frames are positive.

**Parameters to tune.** Window length N; trigger threshold (for the score version).

**Pros.** Trivial to implement and reason about; constant-time update with a running sum
(O(1) amortized); no hidden state beyond the buffer. N is an intuitive "how much history to
trust" knob.

**Cons.** Memory footprint scales with N (buffer of N floats). The window treats all N
frames equally — a stale failure 5 frames ago drags the average down equally to a fresh one,
which dilutes responsiveness to a *new* failure. Requires a warm-up period (fewer than N
frames of history). Binary majority voting discards confidence information, so a few very
high-confidence positives and many near-zero negatives can cancel out.

**Computational cost.** O(1) per frame (running sum), O(N) memory.

**Used in similar applications?** Yes. K-frame majority voting is a standard post-processing
technique for video detection; e.g., construction-site detection for blind/low-vision users
(Feng et al. 2025) found accuracy and F1 improved as K grew from 10 to 50 frames
(0.89→0.91 accuracy, 0.80→0.82 F1), then plateaued, recommending 50-frame majority voting.
Sliding-window mean/voting is pervasive in industrial and surveillance detection.

### 1.2 Exponential smoothing (EWMA / EMA)

**How it works.** Maintain a recursive estimate that weights recent frames more than old
ones, with exponentially decaying weights:

`E_t = α·p_t + (1−α)·E_{t−1}`

where α ∈ (0,1) is the smoothing factor. Trigger when E_t crosses a threshold. This is the
momentum/recursive counterpart to the moving average, and it is the scheme used by the most
widely deployed comparable system (Obico — see §2).

**Parameters to tune.** Smoothing factor α; trigger threshold. Optionally an adaptive
threshold (e.g., trigger when E_t deviates from a rolling baseline by more than k·σ — a
"band"/EWMA-control-chart form).

**Pros.** O(1) time and O(1) memory (single retained value) — no buffer. Recent frames are
weighted more heavily, so it responds faster to a *new* trend while still damping isolated
noise. Smooth, continuous score suitable for an "alert level" meter. A one-line formula;
easy to reason about and tune. Widely used and well-studied.

**Cons.** Selecting α is a trade-off: small α → heavy smoothing, more false-positive
immunity but slower response (misses fast-developing failures); large α → responsive but
noisy (more false positives). With a *fixed* threshold, a slow drift upward that is absorbed
into the baseline can be missed ("the exact class of anomaly you most want to catch — a
slowly worsening failure — gets missed precisely at the settings that look best by
false-positive count alone"; tomato blog EWMA-vs-Kalman study, 2026). Cold-start bias until
the estimate converges. Near the threshold, output can still oscillate/dither frame to
frame, which is why it is usually combined with hysteresis.

**Computational cost.** O(1) time and O(1) memory — the cheapest of the five.

**Used in similar applications?** Yes, prominently. Obico — the leading 3D-print failure
detection system, with >100k users and ~1M failures caught — uses an exponentially weighted
moving average combined with short and long rolling-mean baselines (see §2). EWMA control
charts are a foundational tool in Statistical Process Control (manufacturing) for detecting
process drift (Roberts 1959). The Amazon AWS safety-monitoring reference architecture and
multiple edge-deployed wildfire-smoke detection systems use EMA smoothing of detector
confidence (e.g., α = 0.3–0.4) to reduce false activations. A Prometheus anomaly-detection
comparison found EWMA with an adaptive threshold reached a good detection/false-positive
balance with less parameter search than a Kalman filter — i.e., the *simpler* model was the
easier one to operate.

### 1.3 Consecutive-frame threshold (N-in-a-row)

**How it works.** Require N consecutive frames to all exceed a (binary) positive threshold
before triggering. A single negative resets the counter. Optionally the weaker "K-of-N"
(majority) variant allows up to N−K misses within the window.

**Parameters to tune.** N (consecutive positives required); the per-frame confidence
threshold that defines "positive".

**Pros.** Extremely simple and easy to verify/test. Directly expresses the intuition "a
transient blip must not pause a print." Effectively immune to one-off/flicker false
positives (they never survive N frames). Has a clean interpretation in terms of latency:
trigger attaches at exactly the N-th consecutive positive frame.

**Cons.** A *real* failure whose score intermittently dips below threshold (jitter near the
decision boundary, temporarily occluded by the extruder, motion blur) resets the counter and
delays or entirely misses the trigger — the "K-of-N" variant mitigates this but reintroduces
some false-positive exposure. Wasted history: N−1 tolerated frames are discarded each reset.

**Computational cost.** O(1) time and O(1) memory (a counter).

**Used in similar applications?** Yes. This is a common "multi-frame confirmation" /
"debounce" pattern in industrial inspection and surveillance. iFactory's AI vision
inspection uses "multi-frame confirmation" for borderline detections, requiring two or more
consecutive frames before an auto-reject fires, specifically because a single degraded
frame (motion blur, vibration focus shift) causes false rejects. AWS's safety-monitoring
reference architecture builds a "replication count" of consecutive minutes an object
persists in violation. The nodedb alert library implements per-group consecutive-fire /
consecutive-recover counting. Industrial video inspection guidance routinely recommends
"multi-frame confirmation" to hold below a 2% false-positive rate without losing recall.

### 1.4 Hysteresis (asymmetric entry/exit thresholds)

**How it works.** Maintain a current binary *state* (e.g., FAILING / NOT-FAILING) plus two
thresholds: a high one τ_entry to transition NOT-FAILING→FAILING, and a low one τ_exit to
transition FAILING→NOT-FAILING, with τ_exit < τ_entry. The gap between them creates
"inertia": once triggered, the state persists even if the (smoothed) score dips, and it
stays clear even if the score blips above τ_exit.

**Parameters to tune.** τ_entry, τ_exit, and usually the width of the hysteresis band
(τ_entry − τ_exit); the score signal being thresholded (often already EWMA/moving-averaged).

**Pros.** Eliminates *state chattering* / dithering at a single decision boundary — the
classic "confidence oscillates around 0.5 → toggles every frame" problem. Intuitive and
cheap. Because exit threshold is lower than entry, a brief score dip after triggering does
not clear the alert (avoids "recovery storms" of re-alerts). Well-matched to the monitoring
task where you want the alert to *stay* latched.

**Cons.** Adds state and two thresholds to tune. Hysteresis alone only suppresses
oscillation *around a stable region*; it does not by itself remove isolated false positives
far above τ_entry (that needs smoothing / consecutive confirmation). It is a decision-layer
tool, typically applied *after* a smoothing step.

**Computational cost.** O(1) time and O(1) memory (one retained state + counter of
consecutive above/below).

**Used in similar applications?** Yes, very common in monitoring/control of exactly this
kind (Schmitt-trigger / thermostat analogy; hysteresis between streetlight dark/bright
thresholds). A 2026 vision work-zone detection system combined EMA smoothing (α=0.3) with
asymmetric hysteresis (τ_entry=0.70, τ_exit=0.45) in a finite state machine; against a
YOLO-only baseline it cut false positives 84.6%, reaching F1=0.87 at 2.6 false positives/hour.
The nodedb alert library combines hysteresis *with* consecutive counts. skimage and several
ML libraries ship hysteresis thresholders specifically "for robust decision making in noisy
environments" / "prevents flapping," at O(1) cost.

### 1.5 Finite state machine (FSM)

**How it works.** Model the system as a discrete set of states (e.g., NEVER-STARTED /
OBSERVING / NORMAL / SUSPECT / FAILING / PAUSED) with explicit, rule-defined transitions
between them, each guarded by conditions on the (smoothed) score, frame counts, elapsed
time, or printer state. A single action (pause) is emitted on exactly one transition, so it
fires once per episode rather than repeatedly per frame.

**Parameters to tune.** The state set; the guard conditions for each transition
(thresholds, counts, timings). This is not a *parameter* per se but a design that consumes
the other approaches' thresholds.

**Pros.** The most expressive and robust: it can encode a warm-up window (ignore the first
several minutes, as Obico does), a "suspect/maybe failure" intermediate state (Obico uses a
three-tier no/maybe/yes decision), latching (once FAILING, stay until the print explicitly
ends), fire-once semantics, and recovery rules — all impossible or awkward with a bare
threshold. Makes the decision logic explicit, auditable, and testable (you can unit-test
each transition). Everything is O(1) per frame.

**Cons.** Most complexity to design and validate; more states to reason about and edge
cases to test. Requires discipline to keep the transition graph minimal, or it becomes
unmaintainable. For the MVP it is usually the *wrapper*, not the smoothing, that you add
last.

**Computational cost.** O(1) time and O(1) memory (fixed small state).

**Used in similar applications?** Yes, pervasively — this is the standard way monitoring
systems encode operational logic. Obico's decision logic is effectively a three-tier state
decision over temporal statistics (§2). The 2026 work-zone system is an explicit
OUTSIDE→APPROACHING→INSIDE→EXITING state machine over EMA+hysteresis. A 2026 chemical
inspection system used an "adaptive hysteresis filter" with a state machine whose thresholds
adjust to live noise, reaching F1=93.1% at 1.3 alarms/hour, 24.8 ms/frame on edge
hardware. Alert engines routinely use per-(alert, key) state machines to prevent
re-firing / flapping.

---

## 2. What Obico (and similar systems) actually do

Obico is the most relevant precedent: it is *the* open-source leader in 3D-print webcam
failure detection (>100k users, >89.8M hours monitored, >1M failures caught as of its
published stats). Its decision logic is documented in its Super Detection API and the
obico-server source. It does **not** use an RNN/transformer for the temporal decision —
it uses statistical smoothing over per-frame model outputs:

1. **Per-frame model** → a set of detection boxes with confidence scores.
2. A **normalized per-frame failure probability** `p` is derived (from boxes, sensitivity,
   and calibration).
3. **Temporal statistics** are accumulated per print (returned by the API as `temporal_stats`):
   - `ewm_mean`: an exponentially weighted moving average (alpha = 2/13 ≈ 0.154),
   - `rolling_mean_short`: a short rolling mean (~5 minutes at 10 s/frame ≈ 30 frames),
   - `rolling_mean_long`: a long rolling baseline (~20 hours).
4. **Decision thresholds on the difference** between the EWM and the long baseline:
   - `ewm_mean − rolling_mean_long < 0.36` → **no failure**,
   - `> 0.99` → **failure**,
   - `> 0.78` → **maybe failure**,
   - else if `ewm_mean > (rolling_mean_short − rolling_mean_long) · 3.8` → **maybe failure**.

These "magic numbers" are explicitly documented as hyper-parameters to be tuned. The
comparison of a recent value (EWM) against a *long* baseline ("this print / recent prints
look worse than your normal baseline") is a form of baseline-relative detection that avoids
the "slow drift absorbed into baseline" pitfall noted for a naive fixed-threshold EWMA.

Relevant operational details:
- Obico has a **warm-up**: it deliberately does not alarm in the first several minutes of a
  print, treating high scores then as likely false ("No way! I must be wrong here"). This is
  precisely an FSM state ("OBSERVING") that suppresses alarms early.
- Actions (pause) fire once per print, not per frame.
- Sensitivity is an exposed user control that scales the thresholds — i.e., the 
  false-positive / false-negative trade-off is exposed to the user.

**Bambuddy**, a self-hosted project that uses Obico's ML API, confirms the same math:
"smooths scores over time (Obico's own EWM + rolling-mean math) so a single noisy frame
cannot trigger," with alpha = 2/13, a ~5 min short mean, a ~20 h long mean, and a 30-frame
warmup. It exposes Low/Medium/High sensitivity scaling and action-once-per-print semantics.

**Bambu Lab** printers use a simpler scheme: buffer the latest 10 frames' detection results
and trigger if "enough positive candidates *and they are spatially consistent*" — a
consecutive-frame + spatial-consistency heuristic. So the mainstream deployed system uses
EWMA + rolling means + thresholds + warm-up; a leading hardware vendor uses a 10-frame
accumulation rule.

**Bottom line:** the deployed state of the art in this exact domain is *not* a learned
sequence model — it is statistical smoothing (EWMA + rolling-mean baselines) with tuned
thresholds, a warm-up period, and fire-once action semantics. This is strong evidence that a
simple approach is sufficient and appropriate for the MVP.

---

## 3. Published work on temporal smoothing for anomaly detection in manufacturing

General-purpose video anomaly detection (VAD) research does use learned temporal models —
LSTM, ConvLSTM, GRU, transformers — but these operate *inside* the anomaly model (predicting
future frames / reconstructing normal patterns), not as a decision layer on top of a
frame classifier. Several observations from this literature are directly relevant:

- **The benefit of temporal modeling in VAD comes from learning motion/normality patterns
  in unsupervised reconstruction/prediction**, not from debouncing a supervised
  frame-level classifier. When the task reduces to "classify frames, then decide," the
  literature converges on statistical debouncing of the classifier output (majority voting,
  EMA, hysteresis), not on a second learned network.
- Studies that quantify it (e.g., construction-site K-frame majority voting: accuracy/F1
  improve up to K≈50 then plateau; wildfire EDGE-AI with EMA α=0.4 + voting) show that
  simple temporal aggregation delivers the accuracy benefit. There is **no published case**
  in this review of an RNN/transformer decision layer beating well-tuned
  EWMA + hysteresis + consecutive-count for the *decision* stage of a per-frame classifier.
- Statistical process control literature (EWMA / CUSUM control charts) is the classical,
  extensively validated foundation for detecting drift in manufacturing streams while
  bounding false-alarm rate (Roberts 1959; recent adaptive/truncated EWMA variants directly
  address the "outlier → false alarm" and "slow drift absorbed" failure modes). This is a
  mature body of work directly applicable to the confidence-score stream.
- Dedicated industrial video anomaly detection (IVAD) papers (e.g., MSTE, O-VAD, INFP,
  Industrial-AdaVAD) focus on spatial-temporal *modeling under scarce abnormal data* — a
  different problem (detecting anomalies from reconstruction, no annotated failures) from
  TinyVision's (supervised failure classification + a decision layer). For TinyVision's
  problem, the temporal-decision literature is the simpler, statistical one above.

**Conclusion of the literature review:** the decision-layer problem (smooth a per-frame
score, then threshold with controlled false-positive rate and delay) has a mature, simple,
proven answer in statistical methods. The learned sequence models in VAD address a different
and much harder sub-problem (unsupervised anomaly discovery) that TinyVision does not need
for the failure-classification MVP.

---

## 4. Detection latency vs. false-positive rate trade-off

This is a fundamental, unavoidable trade-off for **any** temporal decision scheme:

- Lowering the trigger threshold / requiring fewer confirming frames ⇒ **lower detection
  latency** (catch failures sooner) but **higher false-positive rate** (more spurious
  pauses).
- Raising the threshold / requiring more confirmation ⇒ **fewer false positives** but
  **longer latency** and a higher chance of missing fast-developing failures (false
  negatives).

The standard framing is a **latency-FPR curve** (analogous to ROC/PRC): for each
per-frame threshold you get a (detection-delay, false-positive-rate) operating point, and
the calibration trade-off is chosen to meet operational requirements (Puccetti & Ceccarelli,
2024/2025; the intrusion/anomaly-detection literature — e.g., "Minimum Delay Object
Detection From Video," ICCV 2019 — formalizes this and shows that minimizing delay for a
given false-alarm constraint is a Quickest-Detection problem). In manufacturing, guidance
(industrial AI inspection) quantifies the human ceiling: operators start overriding a system
at ~3–5% FPR, so deployments target <1–2% FPR; multi-frame confirmation holds that below
1% without loosening recall on true defects.

**For TinyVision's 3D-print case** the trade-off is unusually favorable, for two reasons:

1. **High latency tolerance.** Failures take seconds-to-minutes. At even 5 FPS, a 30- or
   60-frame confirmation window is only 6–12 seconds — far shorter than the timescale of the
   failures being caught. There is nearly free false-positive suppression available: spend
   the "delay budget" to demand confirmation.
2. **Cost asymmetry is tunable and user-exposed.** False positives and false negatives are
   asymmetric, so the operating point must be biased to the false-negative side (as Obico does
   via a user-facing sensitivity control). Because the cost ratio differs per user (an
   overnight unattended print vs. a supervised daytime print), it should be an *exposed*
   setting.

Quantitative guidance for tuning (see §5): choose a confirmation window / EWMA α such that a
spurious high-confidence blip (typically isolated to 1–2 frames) cannot reach the trigger,
while a persistent failure (frames of elevated score) is confirmed within a few seconds. If
the per-frame true-positive rate for a real failure is high (≈1.0 once a failure is clearly
visible), then even a 5-in-a-row rule delays a genuine trigger by ~1 s at 5 FPS — negligible,
while killing nearly all isolated false positives.

---

## 5. Are simple approaches sufficient? (RNN / transformer comparison)

**No clear benefit shown for the decision stage.** The reviewed evidence points the other way:

- The deployed leader in the exact domain (Obico) uses EWMA + rolling means + thresholds +
  warm-up. A major printer vendor (Bambu) uses a 10-frame accumulation + spatial-consistency
  rule. Neither uses a learned sequence model for the decision.
- Multiple studies show statistical temporal aggregation (majority voting up to K≈50; EMA
  α≈0.3–0.4 + hysteresis) reaching the accuracy/FPR targets with O(1) cost (construction,
  wildfire edge-AI, work-zone, industrial inspection).
- The EWMA-vs-Kalman comparison found the *simpler* EWMA the easier to tune for a
  detection/FPR balance, and that naive tuning of the fancier method can *miss gradual
  failures precisely at low-FPR settings* — a pitfall of more complex models, not a benefit.
- An RNN/transformer decision layer over per-frame classifier scores would add: a training
  pipeline and dataset (per-failure temporal sequences), a label-hungry model, higher
  inference cost and latency, less explainability, and more failure modes — for a problem
  that a hand-tuned threshold handles well. It is against the project's principles
  (small models, minimal dependencies, deterministic CLI/logic, reproducibility).

**Where a learned temporal model *might* eventually pay off (not for the MVP):**
if per-frame classification is too noisy to be fixed by debouncing (e.g., many successive
ambiguous frames), or if distinguishing *transient-but-normal* motion from *sustained*
failure requires context beyond a threshold. This remains a research question, not a reason
to build it now.

**Recommendation:** start with **exponential smoothing (EWMA) + an adaptive/baseline-relative
threshold, wrapped in a minimal finite state machine** with **hysteresis** and **fire-once**
semantics. This is the Obico-proven combination, O(1), trivially explainable, and easy to
tune and unit-test. Consecutive-frame and moving-average are useful simpler fallbacks for a
first prototype; the FSM provides the warm-up, "maybe/suspect" intermediate, latching, and
recovery behavior that a bare threshold cannot.

### Suggested MVP decision pipeline

A concrete design to validate during implementation (parameters are starting points, not
final values — tune on held-out captured data, mirroring Obico's documented approach):

1. **Per-frame score** p ∈ [0,1] from the trained classifier/detector.
2. **Smoothed score** via EWMA: `E_t = α·p_t + (1−α)·E_{t−1}`, α ≈ 0.15 (Obico's 2/13;
   tune).
3. **Baseline reference** `B` = long rolling mean of `p` (~minutes-hours) to detect
   "this moment is worse than the print's normal baseline" (avoids the slow-drift pitfall).
4. **FSM states:** `OBSERVING` (warm-up — suppress alarms for the first ~2–5 min,
   per Obico) → `NORMAL` → `SUSPECT` (E_t − B above a low threshold) → `FAILING` (above a
   high threshold, possibly with K-of-N consecutive SUSPECT frames to confirm) → `PAUSED`.
5. **Hysteresis:** enter FAILING above τ_high, only leave below τ_low (τ_low < τ_high),
   preventing chattering and re-alert storms.
6. **Fire-once:** emit the pause/notification on exactly one transition into FAILING per
   print.
7. **Expose sensitivity** as a user setting that scales τ_high/τ_low, biasing toward
   false-negative-minimization by default (per the asymmetric cost).

All parameters (α, baseline window, thresholds, hysteresis gap, K/N, warm-up duration) are
hyper-parameters to be recorded and tuned per the experiment policy.

---

## 6. Approach comparison summary

| Approach | How it works | Parameters | Cost | Pros | Cons | Used in similar apps? |
|---|---|---|---|---|---|---|
| Moving average | mean of last N scores / majority vote | N, threshold | O(1) time, O(N) mem | simple; equal-weight rationale | stale frames dilute response; warm-up; discards confidence if binarized | Yes — K-frame majority voting (construction), industrial/surveillance |
| Exponential smoothing (EWMA) | recursive weighted mean, recent-heavy | α, threshold | O(1) / O(1) | cheapest; responsive; continuous score; widely studied | slow-drift absorption; α is a latency/FPR knob; can dither near threshold | Yes (headline) — Obico, SPC charts, AWS safety ref, wildfire edge-AI |
| Consecutive-frame | N positive frames in a row | N, per-frame threshold | O(1) / O(1) | immune to 1–2 frame blips; clean latency | jitter resets real failures; K-of-N variant re-exposes FPs | Yes — industrial multi-frame confirmation (iFactory, AWS), alert libraries |
| Hysteresis | τ_entry vs τ_exit over current state | τ_entry, τ_exit, band | O(1) / O(1) | kills chattering; latches alerts; avoids re-alert storms | only suppresses boundary oscillation; needs smoothing beneath it | Yes — work-zone (84.6% FP cut), chemical inspection, Schmitt-trigger/thermostat |
| Finite state machine | explicit states + guarded transitions + fire-once | state set, guard params | O(1) / O(1) | warm-up, maybe-state, latching, recovery, testability | most design/validation effort | Yes — Obico (3-tier + warm-up), work-zone, alert engines |

---

## 7. Sources

- Obico Super Detection API docs — `temporal_stats` (ewm_mean, rolling_mean_short/long)
  and threshold decision logic; magic numbers documented as tunable hyper-parameters.
- Obico blog "How Obico AI Failure Detection Works" — capture cadence, sensitivity control,
  no published single accuracy number, FPR/FN behavior by failure type.
- Bambuddy docs — confirms Obico EWM alpha=2/13, ~5 min short mean, ~20 h long mean,
  30-frame warmup, sensitivity scaling, action-once-per-print.
- Bambu Lab chicken/spaghetti detection (per Obico ecosystem docs) — 10-frame buffer +
  spatial-consistency accumulation.
- Feng et al. 2025 (construction-site detection) — K-frame majority voting: gains up to
  K≈50 then plateau.
- 2026 work-zone vision paper — EMA (α=0.3) + hysteresis (0.70/0.45) + FSM: 84.6% FP cut,
  F1 0.87, 2.6 FP/hr.
- iFactory AI vision inspection guides — multi-frame confirmation for borderline detections;
  FPR ceiling 3–5% before operators override; target <1–2%.
- AWS "Automate safety monitoring with computer vision and generative AI" — temporal
  violation tracking / replication count.
- Puccetti & Ceccarelli (2024/2025), "On detection latencies..." — latency-vs-FPR trade-off
  and evaluation methodology (intrusion/error detection).
- Lao & Sundaramoorthi, "Minimum Delay Object Detection From Video," ICCV 2019 — delay/FPR
  as a Quickest-Detection problem.
- EWMA control charts in Statistical Process Control (Roberts 1959; recent adaptive/truncated
  EWMA, e.g., ML-TAE/WMD-EWMA) — drift detection with bounded FPR.
- Prometheus EWMA-vs-Kalman anomaly detection study (2026) — EWMA easier to tune; slow-drift
  pitfall at low-FPR settings.
- Video anomaly detection reviews (2024) and IVAD papers (MSTE, O-VAD, INFP, Industrial-AdaVAD)
  — learned temporal models target unsupervised anomaly discovery, not decision-layer
  debouncing of a frame classifier.

These are research notes for internal decision-making; parameter values should be
re-derived and recorded during implementation, per the experiment policy.
