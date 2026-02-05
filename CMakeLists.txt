# CMakeLists.txt for IBD-Booster
cmake_minimum_required(VERSION 3.15)
project(IBD-Booster LANGUAGES CXX)

# -----------------------------
# C++ Standard
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Include directories
# -----------------------------
include_directories(${PROJECT_SOURCE_DIR}/include)

# -----------------------------
# Source files
# -----------------------------
set(SRC_FILES
    src/pbwt.cpp
    src/vcf_utils.cpp
    src/psmoother.cpp
    src/rPBWT.cpp
    src/fwdPBWT.cpp
    src/feature_extractor.cpp
    src/xgb_predictor.cpp
    main.cpp
)

# -----------------------------
# Main Executable
# -----------------------------
add_executable(IBD-Booster ${SRC_FILES})

# -----------------------------
# Link required libraries
# -----------------------------
include_directories(/data/cchen32/include)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(xgboost REQUIRED)
target_link_libraries(IBD-Booster PRIVATE ZLIB::ZLIB /data/cchen32/lib/libhts.so Threads::Threads OpenMP::OpenMP_CXX xgboost::xgboost)

# -----------------------------
# Compiler optimization flags
# -----------------------------
if(MSVC)
    target_compile_options(IBD-Booster PRIVATE /O2 /DNDEBUG)
else()
    target_compile_options(IBD-Booster PRIVATE -O3 -march=native -flto -funroll-loops -DNDEBUG)
    target_link_options(IBD-Booster PRIVATE -flto)
endif()
