# TinyVision edge inference runtime
#
# Build outputs:
#   * bin/tinyvision        - CLI application
#   * bin/tinyvision_tests  - focused parity/unit test binary (CTest)
#
# Dependencies (Homebrew on macOS; apt packages on Raspberry Pi):
#   * OpenCV (>= 4.x / 5.x)
#   * ONNX Runtime (C++ headers + shared library)
#
# Configure/run:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#   ./build/bin/tinyvision --input video.mp4
#
# Runtime target: Raspberry Pi 4/5 + USB/CSI webcam (CPU; ONNX Runtime CPU EP).
# The Python `tv poc` pipeline is the behavioral reference.
cmake_minimum_required(VERSION 3.16)
project(tinyvision_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

# ONNX Runtime ---------------------------------------------------------------
# The official prebuilt tarballs (e.g. onnxruntime-linux-x64-<ver>.tgz) ship a
# CMake package config, but it hardcodes a `lib64/` libdir that the tarball does
# not contain, so `find_package(onnxruntime)` fails against them. Locate the
# header and shared library directly first (rooted at -Donnxruntime_ROOT /
# -Donnxruntime_DIR / $ONNXRUNTIME_ROOT, plus common prefixes), and fall back to
# a package config for Homebrew / vcpkg / source installs.
set(_onnxruntime_hints)
foreach(_onnxruntime_root_var onnxruntime_ROOT ONNXRUNTIME_ROOT)
  if(DEFINED ${_onnxruntime_root_var})
    list(APPEND _onnxruntime_hints "${${_onnxruntime_root_var}}")
  endif()
  if(DEFINED ENV{${_onnxruntime_root_var}})
    list(APPEND _onnxruntime_hints "$ENV{${_onnxruntime_root_var}}")
  endif()
endforeach()
if(DEFINED onnxruntime_DIR)
  get_filename_component(_onnxruntime_prefix "${onnxruntime_DIR}" DIRECTORY)
  list(APPEND _onnxruntime_hints "${_onnxruntime_prefix}" "${onnxruntime_DIR}")
endif()
if(APPLE)
  list(APPEND _onnxruntime_hints /opt/homebrew /usr/local /opt/local)
endif()

find_path(onnxruntime_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  HINTS ${_onnxruntime_hints}
  PATH_SUFFIXES include include/onnxruntime
)
find_library(onnxruntime_LIBRARY
  NAMES onnxruntime
  HINTS ${_onnxruntime_hints}
  PATH_SUFFIXES lib lib64
)

if(onnxruntime_INCLUDE_DIR AND onnxruntime_LIBRARY)
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}")
  endif()
else()
  find_package(onnxruntime REQUIRED CONFIG)
endif()

add_library(tinyvision_core STATIC
  src/cli.cpp
  src/preprocess.cpp
  src/model.cpp
  src/temporal.cpp
)
target_include_directories(tinyvision_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tinyvision_core
  PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime
)

add_executable(tinyvision src/main.cpp)
target_link_libraries(tinyvision PRIVATE tinyvision_core)

enable_testing()

add_executable(tinyvision_tests
  tests/harness.cpp
  tests/test_preprocess.cpp
  tests/test_temporal.cpp
  tests/test_model.cpp
  tests/test_cli.cpp
)
target_link_libraries(tinyvision_tests PRIVATE tinyvision_core)
# Tests read golden files and the ONNX model via repo-root-relative paths.
set_property(TARGET tinyvision_tests PROPERTY CXX_STANDARD 17)
enable_testing()
add_test(NAME tinyvision_tests COMMAND tinyvision_tests)
set_tests_properties(tinyvision_tests PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
add_test(NAME integration_smoke COMMAND bash runtime/tests/integration_smoke.sh)
set_tests_properties(integration_smoke PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)