// Temporal decision logic: faithful port of training/tv/temporal.py.
//
//   per-frame P(failing) -> EWMA smoothing (EwmaBaseline) ->
//   FSM (OBSERVING -> NORMAL -> SUSPECT -> FAILING -> PAUSED) with hysteresis
//   and fire-once semantics.
//
// The arithmetic uses doubles in exactly the same order as the Python
// reference so that, given identical input scores, the outputs (signal,
// baseline, delta, state, event) match to floating-point precision.
#pragma once

#include <string>
#include <vector>

#include "tinyvision/config.hpp"

namespace tinyvision {

inline constexpr const char* kFailingEvent = "failing_event";

enum class FsmState : int {
  kObserving = 0,
  kNormal = 1,
  kSuspect = 2,
  kFailing = 3,
  kPaused = 4,
};

std::string StateName(FsmState s);

struct EwmaStep {
  double signal = 0.0;
  double baseline = 0.0;
  double delta = 0.0;
};

class EwmaBaseline {
 public:
  EwmaBaseline(double alpha, int baseline_window);
  EwmaStep Step(double p);
  void Reset();

 private:
  double alpha_;
  int window_;
  double sig_ = 0.0;
  std::vector<double> buf_;
};

struct FsmDecision {
  FsmState state = FsmState::kObserving;
  double signal = 0.0;
  double delta = 0.0;
  int suspect_count = 0;
  bool event = false;  // true exactly once on the NORMAL/SUSPECT -> FAILING edge
  int frame = 0;
};

class FailureFsm {
 public:
  explicit FailureFsm(const TemporalParams& p);
  FsmDecision Step(double signal, double baseline);
  void Reset();

 private:
  void EnterFailing(FsmDecision& d);
  TemporalParams p_;
  FsmState state_ = FsmState::kObserving;
  int frame_ = 0;
  int suspect_count_ = 0;
  bool fired_ = false;
};

// Per-frame decision row, mirroring the POC's DecisionTracker output keys.
struct FrameDecision {
  int frame = 0;       // 0-based sampled-frame index (matches POC "frame")
  int raw_frame = 0;   // 1-based raw video frame index (matches POC "frame_idx")
  double raw = 0.0;
  double signal = 0.0;
  double baseline = 0.0;
  double delta = 0.0;
  std::string state;        // state name exactly as in Python
  bool raw_failing = false; // raw fail_score >= threshold
  std::string event;        // "" or "failing_event"
};

class DecisionTracker {
 public:
  explicit DecisionTracker(const TemporalParams& p, double threshold);
  FrameDecision Step(double fail_score);
  void Reset();
  const std::vector<FrameDecision>& rows() const { return rows_; }
  // Record the raw video frame index for an already-pushed decision row.
  void SetFrameIdx(int sampled_frame, int raw_frame) {
    rows_[sampled_frame].raw_frame = raw_frame;
  }

 private:
  EwmaBaseline ew_;
  FailureFsm fsm_;
  double threshold_;
  std::vector<FrameDecision> rows_;
};

}  // namespace tinyvision