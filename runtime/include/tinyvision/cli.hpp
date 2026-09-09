// CLI argument parsing for the edge runtime.
//
// All deployment-relevant parameters are configurable via flags; the defaults
// (config.hpp) are the validated POC defaults and are preserved here.
#pragma once

#include <string>

#include "tinyvision/config.hpp"

namespace tinyvision {

struct Cli {
  RuntimeConfig cfg;
  std::string help;  // non-empty => --help/-h requested
  bool used_input = false;
  bool used_camera = false;
};

// Parse argv (excluding argv[0]). Throws std::runtime_error on malformed
// configuration (unknown option, missing value, mutually exclusive or invalid
// combination). Help does not throw.
Cli ParseArgs(int argc, char** argv);

}  // namespace tinyvision