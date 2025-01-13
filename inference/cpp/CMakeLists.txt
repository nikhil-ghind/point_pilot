cmake_minimum_required(VERSION 3.16)
project(PointPillarsInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── Options ──────────────────────────────────────────────────────────────────

option(WITH_OPENCV "Build with OpenCV visualization support" OFF)

# Path to ONNX Runtime installation
# Override with: cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root directory")

# ── Find ONNX Runtime ─────────────────────────────────────────────────────────

if(ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
    )
else()
    # Try to find system-installed ONNX Runtime
    find_path(ONNXRUNTIME_INCLUDE_DIRS
        NAMES onnxruntime/core/session/onnxruntime_cxx_api.h
        PATHS
            /usr/local/include
            /usr/include
            /opt/onnxruntime/include
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            /usr/local/lib
            /usr/lib
            /opt/onnxruntime/lib
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to the installation directory.\n"
        "Example: cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.16.3\n"
        "Download from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()

message(STATUS "ONNX Runtime includes: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library:  ${ONNXRUNTIME_LIB}")

# ── Optional: OpenCV ──────────────────────────────────────────────────────────

if(WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()

# ── Target ───────────────────────────────────────────────────────────────────

add_executable(pointpillars_infer
    pointpillars_infer.cpp
    voxelizer.cpp
)

target_include_directories(pointpillars_infer PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pointpillars_infer PRIVATE
    ${ONNXRUNTIME_LIB}
)

if(WITH_OPENCV)
    target_link_libraries(pointpillars_infer PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(pointpillars_infer PRIVATE WITH_OPENCV)
endif()

# Compiler flags
target_compile_options(pointpillars_infer PRIVATE
    -O3
    -Wall
    -Wextra
    -march=native
)

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS pointpillars_infer
    RUNTIME DESTINATION bin
)

# ── Summary ───────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== PointPillars Inference Build Config ===")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:       ${WITH_OPENCV}")
message(STATUS "==========================================")
message(STATUS "")
message(STATUS "Build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
message(STATUS "  make -j4")
message(STATUS "")
message(STATUS "Run:")
message(STATUS "  ./pointpillars_infer model.onnx pointcloud.bin [score_threshold]")
