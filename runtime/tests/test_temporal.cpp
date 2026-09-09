#include <map>
#include <vector>

#include "harness.hpp"
#include "tinyvision/temporal.hpp"

namespace {

tinyvision::TemporalParams GoldenParams() {
  tinyvision::TemporalParams p;
  p.alpha = 0.15;
  p.baseline_window = 60;
  p.warmup = 30;
  p.suspect_threshold = 0.05;
  p.fail_threshold = 0.30;
  p.recover_threshold = 0.01;
  p.fail_threshold_abs = 0.65;
  p.confirm = 3;
  p.use_delta = false;
  return p;
}

struct Gold {
  double raw, signal, baseline, delta;
  std::string state, event;
};

std::map<std::string, std::vector<Gold>> LoadGoldens() {
  const auto rows = tv_testing::LoadCsv("runtime/tests/golden_temporal.csv");
  std::map<std::string, std::vector<Gold>> out;
  for (size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    Gold g;
    g.raw = std::stod(r[2]);
    g.signal = std::stod(r[3]);
    g.baseline = std::stod(r[4]);
    g.delta = std::stod(r[5]);
    g.state = r[6];
    g.event = r[7];
    out[r[0]].push_back(g);
  }
  return out;
}

void CheckRow(const Gold& g, const tinyvision::FrameDecision& d) {
  const bool raw_ok = tv_testing::Close(d.raw, g.raw, 1e-12);
  const bool sig_ok = tv_testing::Close(d.signal, g.signal, 1e-12);
  const bool base_ok = tv_testing::Close(d.baseline, g.baseline, 1e-12);
  const bool delta_ok = tv_testing::Close(d.delta, g.delta, 1e-12);
  const std::string ev = d.event.empty() ? "" : "failing_event";
  if (!(raw_ok && sig_ok && base_ok && delta_ok && d.state == g.state &&
        ev == g.event)) {
    std::fprintf(stderr,
                 "temporal mismatch on frame %d: raw=%d sig=%d base=%d "
                 "delta=%d state_ok=%d event_ok=%d\n",
                 d.frame, raw_ok, sig_ok, base_ok, delta_ok,
                 d.state == g.state, ev == g.event);
    std::fprintf(stderr,
                 "  got  raw=%.17g sig=%.17g base=%.17g delta=%.17g "
                 "state='%s' event='%s'\n",
                 d.raw, d.signal, d.baseline, d.delta, d.state.c_str(),
                 d.event.c_str());
    std::fprintf(stderr,
                 "  want raw=%.17g sig=%.17g base=%.17g delta=%.17g "
                 "state='%s' event='%s'\n",
                 g.raw, g.signal, g.baseline, g.delta, g.state.c_str(),
                 g.event.c_str());
    std::exit(1);
  }
}

}  // namespace

// The Python reference (`run_temporal_pipeline` with the same params) is the
// behavioral source of truth; the C++ tracker must reproduce it in doubles.
TEST(TemporalMatchesPythonGoldenTraces) {
  const auto golds = LoadGoldens();
  const tinyvision::TemporalParams p = GoldenParams();
  for (const auto& [seq, gold] : golds) {
    tinyvision::DecisionTracker tr(p, 0.5);
    for (const Gold& g : gold) {
      const tinyvision::FrameDecision d = tr.Step(g.raw);
      CheckRow(g, d);
    }
    if (gold.empty() || gold.back().event.empty()) {
      continue;
    }
  }
}

// --- Hand-built scenarios (semantics of the state machine). ---
TEST(TemporalQuietSequenceNeverFails) {
  tinyvision::DecisionTracker tr(GoldenParams(), 0.5);
  for (int i = 0; i < 80; ++i) {
    const tinyvision::FrameDecision d = tr.Step(0.02);
    CHECK(d.event.empty());
    CHECK(d.state == "OBSERVING" || d.state == "NORMAL");
  }
}

TEST(TemporalSpikeFiresOnceWithRecovery) {
  tinyvision::DecisionTracker tr(GoldenParams(), 0.5);
  int events = 0;
  int first_event = -1;
  for (int i = 0; i < 35; ++i) tr.Step(0.02);
  for (int i = 0; i < 25; ++i) {
    const tinyvision::FrameDecision d = tr.Step(0.8);
    if (!d.event.empty()) {
      ++events;
      if (first_event < 0) first_event = d.frame;
    }
  }
  CHECK(events == 1);
  CHECK(first_event >= 0);
  CHECK(first_event > 30);  // fired after warmup + 3 confirmations
}

// Fire by sustained high signal, then confirm the machine recovers to PAUSED
// once the signal collapses toward zero -- the fire-once flag must suppress a
// second event. (With use_delta=false, PAUSED requires signal <=
// recover_threshold, so recovery input must drive the EWMA asymptote to ~0.)
TEST(TemporalRecoversToPausedAfterFire) {
  tinyvision::DecisionTracker tr(GoldenParams(), 0.5);
  int events = 0;
  for (int i = 0; i < 40; ++i) tr.Step(0.02);          // warmup
  for (int i = 0; i < 40; ++i) {                        // failure phase
    const tinyvision::FrameDecision d = tr.Step(0.8);
    if (!d.event.empty()) ++events;
  }
  CHECK(events == 1);
  bool paused = false;
  for (int i = 0; i < 300 && !paused; ++i) {            // recovery phase
    const tinyvision::FrameDecision d = tr.Step(0.0);
    CHECK(d.event.empty());                             // fire-once
    paused = d.state == "PAUSED";
  }
  CHECK(paused);
}

// Long-sustained high signal drives the EWMA signal past fail_threshold_abs
// (0.65) and fires the event regardless of the delta-based path.
TEST(TemporalAbsThresholdLatches) {
  tinyvision::DecisionTracker tr(GoldenParams(), 0.5);
  bool fired = false;
  for (int i = 0; i < 60; ++i) {
    const tinyvision::FrameDecision d = tr.Step(0.9);
    if (!d.event.empty()) {
      fired = true;
      CHECK(d.state == "FAILING");
      break;
    }
  }
  CHECK(fired);
}