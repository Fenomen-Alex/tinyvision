#include "tinyvision/temporal.hpp"

#include <algorithm>
#include <stdexcept>

namespace tinyvision {

std::string StateName(FsmState s) {
  switch (s) {
    case FsmState::kObserving:
      return "OBSERVING";
    case FsmState::kNormal:
      return "NORMAL";
    case FsmState::kSuspect:
      return "SUSPECT";
    case FsmState::kFailing:
      return "FAILING";
    case FsmState::kPaused:
      return "PAUSED";
  }
  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// EwmaBaseline
// ---------------------------------------------------------------------------

EwmaBaseline::EwmaBaseline(double alpha, int baseline_window)
    : alpha_(alpha), window_(baseline_window) {
  if (!(alpha_ > 0.0 && alpha_ <= 1.0)) {
    throw std::invalid_argument("alpha must be in (0,1]");
  }
}

void EwmaBaseline::Reset() {
  sig_ = 0.0;
  buf_.clear();
}

EwmaStep EwmaBaseline::Step(double p) {
  sig_ = alpha_ * p + (1.0 - alpha_) * sig_;
  buf_.push_back(sig_);
  if (static_cast<int>(buf_.size()) > window_) {
    buf_.erase(buf_.begin());
  }
  double sum = 0.0;
  for (double v : buf_) {
    sum += v;
  }
  EwmaStep s;
  s.signal = sig_;
  s.baseline = sum / static_cast<double>(buf_.size());
  s.delta = s.signal - s.baseline;
  return s;
}

// ---------------------------------------------------------------------------
// FailureFsm -- port of FailureFsm from training/tv/temporal.py.
// ---------------------------------------------------------------------------

FailureFsm::FailureFsm(const TemporalParams& p) : p_(p) {
  if (p_.fail_threshold_abs != 0.0 && !(p_.fail_threshold_abs >= 0.0 &&
                                        p_.fail_threshold_abs <= 1.0)) {
    throw std::invalid_argument("fail_threshold_abs must be in [0,1] or 0 (unset)");
  }
  if (p_.recover_threshold >= p_.fail_threshold) {
    throw std::invalid_argument(
        "recover_threshold must be < fail_threshold (hysteresis)");
  }
}

void FailureFsm::Reset() {
  state_ = FsmState::kObserving;
  frame_ = 0;
  suspect_count_ = 0;
  fired_ = false;
}

double GuardValue(const TemporalParams& p, const FsmDecision& d) {
  return p.use_delta ? d.delta : d.signal;
}

void FailureFsm::EnterFailing(FsmDecision& d) {
  const bool transitioning =
      state_ == FsmState::kNormal || state_ == FsmState::kSuspect;
  state_ = FsmState::kFailing;
  if (transitioning && !fired_) {
    d.event = true;
    fired_ = true;
  }
}

FsmDecision FailureFsm::Step(double signal, double baseline) {
  ++frame_;
  const double delta = signal - baseline;

  FsmDecision d;
  d.state = state_;
  d.signal = signal;
  d.delta = delta;
  d.suspect_count = suspect_count_;
  d.frame = frame_;

  const double guard = p_.use_delta ? delta : signal;

  if (state_ != FsmState::kPaused) {
    if (state_ == FsmState::kObserving) {
      if (frame_ >= p_.warmup) {
        state_ = FsmState::kNormal;
      }
    } else if (state_ == FsmState::kFailing) {
      if (guard <= p_.recover_threshold) {
        state_ = FsmState::kPaused;
      }
    } else {
      // NORMAL / SUSPECT
      const bool abs_hit = p_.fail_threshold_abs > 0.0 &&
                           signal >= p_.fail_threshold_abs;
      if (abs_hit) {
        EnterFailing(d);
      } else if (guard >= p_.fail_threshold) {
        ++suspect_count_;
        if (suspect_count_ >= p_.confirm) {
          EnterFailing(d);
        } else if (state_ != FsmState::kSuspect) {
          state_ = FsmState::kSuspect;
        }
      } else if (state_ == FsmState::kSuspect) {
        if (guard <= p_.suspect_threshold) {
          suspect_count_ = 0;
          state_ = FsmState::kNormal;
        }
      } else if (guard >= p_.suspect_threshold) {
        state_ = FsmState::kSuspect;
        suspect_count_ = 1;
      }
    }
  }

  d.state = state_;
  d.suspect_count = suspect_count_;
  return d;
}

// ---------------------------------------------------------------------------
// DecisionTracker -- mirrors the POC's DecisionTracker.
// ---------------------------------------------------------------------------

DecisionTracker::DecisionTracker(const TemporalParams& p, double threshold)
    : ew_(p.alpha, p.baseline_window), fsm_(p), threshold_(threshold) {}

void DecisionTracker::Reset() {
  ew_.Reset();
  fsm_.Reset();
  rows_.clear();
}

FrameDecision DecisionTracker::Step(double fail_score) {
  const EwmaStep st = ew_.Step(fail_score);
  const FsmDecision d = fsm_.Step(st.signal, st.baseline);
  FrameDecision r;
  r.frame = static_cast<int>(rows_.size());
  r.raw = fail_score;
  r.signal = st.signal;
  r.baseline = st.baseline;
  r.delta = st.delta;
  r.state = StateName(d.state);
  r.raw_failing = fail_score >= threshold_;
  r.event = d.event ? std::string(kFailingEvent) : std::string();
  rows_.push_back(r);
  return r;
}

}  // namespace tinyvision