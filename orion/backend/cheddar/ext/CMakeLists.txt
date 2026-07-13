# Build the pybind11 extension that wraps Cheddar's CKKS API.
#
# Usage:
#   cmake -S . -B build \
#       -D CHEDDAR_SOURCE_DIR=/path/to/cheddar \
#       -D CHEDDAR_BUILD_DIR=/path/to/cheddar/build \
#       -D CMAKE_CUDA_ARCHITECTURES=86 \
#       -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
#   cmake --build build -j 8
#
# Output: _cheddar_native.cpython-<py>-<plat>.so dropped one dir up
# (in orion/backend/cheddar/), so `from . import _cheddar_native` works
# from bindings.py.
#
# Cheddar's own CMake doesn't install/export a package config; we link
# against the built libcheddar.so and pull headers from the source tree.
# Header deps: rmm (from cheddar's FetchContent _deps) + spdlog (rmm
# 22.12 requirement; conda base provides it -- see SPDLOG_INCLUDE_DIR).

cmake_minimum_required(VERSION 3.24)
project(orion_cheddar_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # baahl1: dual RTX 3090 (sm_86). Override -D for H100 (90), A100 (80).
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT DEFINED CHEDDAR_SOURCE_DIR)
    message(FATAL_ERROR "Set -D CHEDDAR_SOURCE_DIR=/path/to/cheddar")
endif()
if(NOT DEFINED CHEDDAR_BUILD_DIR)
    set(CHEDDAR_BUILD_DIR "${CHEDDAR_SOURCE_DIR}/build")
endif()
if(NOT EXISTS "${CHEDDAR_BUILD_DIR}/libcheddar.so")
    message(FATAL_ERROR
        "libcheddar.so not found in ${CHEDDAR_BUILD_DIR}. Build Cheddar "
        "first (see orion/backend/cheddar/README.md).")
endif()

# rmm headers come from Cheddar's own FetchContent checkout so versions
# always match what libcheddar.so was built against.
set(RMM_INCLUDE_DIR "${CHEDDAR_BUILD_DIR}/_deps/rmm-src/include")
if(NOT EXISTS "${RMM_INCLUDE_DIR}/rmm/device_uvector.hpp")
    message(FATAL_ERROR "rmm headers not found at ${RMM_INCLUDE_DIR}")
endif()

# spdlog: rmm 22.12 includes spdlog headers. Allow override; default to
# the conda base env that satisfied Cheddar's own configure.
if(NOT DEFINED SPDLOG_INCLUDE_DIR)
    set(SPDLOG_INCLUDE_DIR "$ENV{HOME}/miniconda3/include")
endif()

find_package(CUDAToolkit 11.8 REQUIRED)

# ---------------------------------------------------------------------------
# pybind11 (via pip in the active Python environment)
# ---------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_status
)
if(NOT _pybind11_status EQUAL 0)
    message(FATAL_ERROR
        "Couldn't locate pybind11 via the active Python. "
        "Install with: pip install pybind11")
endif()

find_package(pybind11 REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------

pybind11_add_module(_cheddar_native cheddar_pybind.cu)

target_include_directories(_cheddar_native PRIVATE
    "${CHEDDAR_SOURCE_DIR}/include"
    "${RMM_INCLUDE_DIR}"
    "${SPDLOG_INCLUDE_DIR}")

# Conda's spdlog is compiled against an external fmt (no bundled copy),
# matching how Cheddar itself was configured.
target_compile_definitions(_cheddar_native PRIVATE SPDLOG_FMT_EXTERNAL)

target_link_libraries(_cheddar_native PRIVATE
    "${CHEDDAR_BUILD_DIR}/libcheddar.so"
    CUDA::cudart)

set_target_properties(_cheddar_native PROPERTIES
    # Find libcheddar.so at runtime without LD_LIBRARY_PATH.
    BUILD_RPATH "${CHEDDAR_BUILD_DIR}"
    INSTALL_RPATH "${CHEDDAR_BUILD_DIR}"
    # Drop the .so alongside bindings.py (one level up).
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/..")
