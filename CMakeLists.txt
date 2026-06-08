cmake_minimum_required(VERSION 3.18)
project(IBD-Booster LANGUAGES CXX)

include(ExternalProject)
include(GNUInstallDirs)
include(CheckIPOSupported)

# -----------------------------
# Guard for CMake version
# -----------------------------
if(${CMAKE_VERSION} VERSION_LESS "3.18")
    message(FATAL_ERROR
        "CMake >= 3.18 is required.\n"
        "Detected version: ${CMAKE_VERSION}"
    )
endif()

# -----------------------------
# Options
# -----------------------------
option(USE_SYSTEM_ZLIB       "Use system-installed ZLIB if available" ON)
option(USE_SYSTEM_HTSLIB     "Use system-installed HTSlib if available" ON)
option(USE_SYSTEM_XGBOOST    "Use system-installed XGBoost if available" OFF)
option(ENABLE_OPENMP         "Enable OpenMP if available" ON)

# -----------------------------
# C++ Standard
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    src/nn_predictor.cpp
    src/xgb_predictor.cpp
    main.cpp
)

add_executable(IBD-Booster ${SRC_FILES})
target_include_directories(IBD-Booster PRIVATE ${PROJECT_SOURCE_DIR}/include)

# -----------------------------
# Threads
# -----------------------------
find_package(Threads REQUIRED)
target_link_libraries(IBD-Booster PRIVATE Threads::Threads)

# -----------------------------
# OpenMP
# -----------------------------
set(HAVE_OPENMP FALSE)
if(ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(HAVE_OPENMP TRUE)
        target_link_libraries(IBD-Booster PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP found and enabled")
    else()
        message(STATUS "OpenMP not found; building without OpenMP")
    endif()
endif()

# =========================================================
# ZLIB
# =========================================================
set(HAVE_ZLIB FALSE)

if(USE_SYSTEM_ZLIB)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        set(HAVE_ZLIB TRUE)
        message(STATUS "Using system ZLIB")
    else()
        set(USE_SYSTEM_ZLIB OFF CACHE BOOL "Use system-installed ZLIB if available" FORCE)
        message(STATUS "System ZLIB not found; automatically falling back to bundled ZLIB")
    endif()
endif()

if(NOT HAVE_ZLIB)
    message(STATUS "Building bundled ZLIB (static) from GitHub release")

    set(ZLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/zlib-install)
    file(MAKE_DIRECTORY ${ZLIB_INSTALL_DIR}/include)
    file(MAKE_DIRECTORY ${ZLIB_INSTALL_DIR}/lib)

    ExternalProject_Add(zlib_external
        URL https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz
        PREFIX ${CMAKE_BINARY_DIR}/zlib_build
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )

    add_library(ZLIB::ZLIB STATIC IMPORTED GLOBAL)
    set_target_properties(ZLIB::ZLIB PROPERTIES
        IMPORTED_LOCATION ${ZLIB_INSTALL_DIR}/lib/libz.a
        INTERFACE_INCLUDE_DIRECTORIES ${ZLIB_INSTALL_DIR}/include
    )
    add_dependencies(ZLIB::ZLIB zlib_external)
endif()

target_link_libraries(IBD-Booster PRIVATE ZLIB::ZLIB)

# =========================================================
# HTSlib
# =========================================================
set(HAVE_HTSLIB FALSE)

if(USE_SYSTEM_HTSLIB)
    find_path(HTSLIB_INCLUDE_DIR NAMES htslib/hts.h)
    find_library(HTSLIB_LIBRARY NAMES hts libhts)

    if(HTSLIB_INCLUDE_DIR AND HTSLIB_LIBRARY)
        set(HAVE_HTSLIB TRUE)
        add_library(htslib::hts UNKNOWN IMPORTED GLOBAL)
        set_target_properties(htslib::hts PROPERTIES
            IMPORTED_LOCATION ${HTSLIB_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${HTSLIB_INCLUDE_DIR}
        )
        message(STATUS "Using system HTSlib")
    else()
        set(USE_SYSTEM_HTSLIB OFF CACHE BOOL "Use system-installed HTSlib if available" FORCE)
        message(STATUS "System HTSlib not found; automatically falling back to bundled HTSlib")
    endif()
endif()

if(NOT HAVE_HTSLIB)
    message(STATUS "Building bundled HTSlib")

    set(HTSLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/htslib-install)
    file(MAKE_DIRECTORY ${HTSLIB_INSTALL_DIR}/include)
    file(MAKE_DIRECTORY ${HTSLIB_INSTALL_DIR}/lib)

    ExternalProject_Add(htslib_external
        URL https://github.com/samtools/htslib/releases/download/1.20/htslib-1.20.tar.bz2
        PREFIX ${CMAKE_BINARY_DIR}/htslib_build
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND
            <SOURCE_DIR>/configure
            --prefix=${HTSLIB_INSTALL_DIR}
            --disable-bz2
            --disable-lzma
            --disable-libcurl
            --disable-s3
            --disable-gcs
            --without-libdeflate
            --disable-plugins
        BUILD_COMMAND make
        INSTALL_COMMAND make install
        BUILD_IN_SOURCE 1
    )

    add_library(htslib::hts STATIC IMPORTED GLOBAL)
    set_target_properties(htslib::hts PROPERTIES
        IMPORTED_LOCATION ${HTSLIB_INSTALL_DIR}/lib/libhts.a
        INTERFACE_INCLUDE_DIRECTORIES ${HTSLIB_INSTALL_DIR}/include
    )
    add_dependencies(htslib::hts htslib_external)
endif()

target_link_libraries(IBD-Booster PRIVATE htslib::hts)

# =========================================================
# XGBoost
# =========================================================
set(HAVE_XGBOOST FALSE)
set(XGBOOST_IS_SYSTEM FALSE)
set(XGBOOST_IS_BUNDLED_STATIC FALSE)

if(USE_SYSTEM_XGBOOST)
    find_package(xgboost CONFIG QUIET)
    if(TARGET xgboost::xgboost)
        set(HAVE_XGBOOST TRUE)
        set(XGBOOST_IS_SYSTEM TRUE)
        message(STATUS "Using system XGBoost via xgboost::xgboost")
    elseif(TARGET xgboost)
        add_library(xgboost::xgboost ALIAS xgboost)
        set(HAVE_XGBOOST TRUE)
        set(XGBOOST_IS_SYSTEM TRUE)
        message(STATUS "Using system XGBoost via xgboost target")
    else()
        set(USE_SYSTEM_XGBOOST OFF CACHE BOOL "Use system-installed XGBoost if available" FORCE)
        message(STATUS "System XGBoost not found; automatically falling back to bundled static XGBoost")
    endif()
endif()

if(NOT HAVE_XGBOOST)
    message(STATUS "Building bundled XGBoost (static)")

    set(XGBOOST_INSTALL_DIR ${CMAKE_BINARY_DIR}/xgboost-install)
    set(XGBOOST_BUILD_DIR   ${CMAKE_BINARY_DIR}/xgboost_build)
    set(XGBOOST_BUILD_SUBDIR ${XGBOOST_BUILD_DIR}/src/xgboost_external-build)

    file(MAKE_DIRECTORY ${XGBOOST_INSTALL_DIR}/include)
    file(MAKE_DIRECTORY ${XGBOOST_INSTALL_DIR}/lib)

    set(XGBOOST_USE_OPENMP OFF)
    if(HAVE_OPENMP)
        set(XGBOOST_USE_OPENMP ON)
    endif()

    ExternalProject_Add(xgboost_external
        URL https://github.com/dmlc/xgboost/releases/download/v3.2.0/xgboost-src-3.2.0.tar.gz
        PREFIX ${XGBOOST_BUILD_DIR}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${XGBOOST_INSTALL_DIR}
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_STATIC_LIB=ON
            -DUSE_CUDA=OFF
            -DUSE_OPENMP=${XGBOOST_USE_OPENMP}
            -DR_LIB=OFF
            -DJVM_BINDINGS=OFF
            -DPLUGIN_FEDERATED=OFF
            -DGOOGLE_TEST=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        BUILD_BYPRODUCTS
            ${XGBOOST_INSTALL_DIR}/lib/libxgboost.a
            ${XGBOOST_BUILD_SUBDIR}/dmlc-core/libdmlc.a
    )

    add_library(xgboost::xgboost STATIC IMPORTED GLOBAL)
    set_target_properties(xgboost::xgboost PROPERTIES
        IMPORTED_LOCATION ${XGBOOST_INSTALL_DIR}/lib/libxgboost.a
        INTERFACE_INCLUDE_DIRECTORIES ${XGBOOST_INSTALL_DIR}/include
    )
    add_dependencies(xgboost::xgboost xgboost_external)

    add_library(xgboost::dmlc STATIC IMPORTED GLOBAL)
    set_target_properties(xgboost::dmlc PROPERTIES
        IMPORTED_LOCATION ${XGBOOST_BUILD_SUBDIR}/dmlc-core/libdmlc.a
    )
    add_dependencies(xgboost::dmlc xgboost_external)

    set(HAVE_XGBOOST TRUE)
    set(XGBOOST_IS_BUNDLED_STATIC TRUE)
endif()

if(XGBOOST_IS_SYSTEM)
    target_link_libraries(IBD-Booster PRIVATE xgboost::xgboost)
else()
    target_link_libraries(IBD-Booster PRIVATE
        xgboost::xgboost
        xgboost::dmlc
    )
endif()

# Linux extras
if(UNIX AND NOT APPLE)
    target_link_libraries(IBD-Booster PRIVATE dl m)
endif()

# No private runtime lib dir, since bundled XGBoost is static here.
set_target_properties(IBD-Booster PROPERTIES
    BUILD_RPATH ""
    INSTALL_RPATH ""
)

# =========================================================
# Optimization
# =========================================================
if(NOT MSVC)
    target_compile_options(IBD-Booster PRIVATE -O3 -march=native)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set_property(TARGET IBD-Booster PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# =========================================================
# Install
# =========================================================
install(TARGETS IBD-Booster
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# =========================================================
# Summary
# =========================================================
message(STATUS "==== Build Summary ====")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "ENABLE_OPENMP: ${ENABLE_OPENMP}")
message(STATUS "USE_SYSTEM_ZLIB: ${USE_SYSTEM_ZLIB}")
message(STATUS "USE_SYSTEM_HTSLIB: ${USE_SYSTEM_HTSLIB}")
message(STATUS "USE_SYSTEM_XGBOOST: ${USE_SYSTEM_XGBOOST}")
message(STATUS "HAVE_OPENMP: ${HAVE_OPENMP}")
message(STATUS "HAVE_ZLIB: ${HAVE_ZLIB}")
message(STATUS "HAVE_HTSLIB: ${HAVE_HTSLIB}")
message(STATUS "HAVE_XGBOOST: ${HAVE_XGBOOST}")
message(STATUS "XGBOOST_IS_SYSTEM: ${XGBOOST_IS_SYSTEM}")
message(STATUS "XGBOOST_IS_BUNDLED_STATIC: ${XGBOOST_IS_BUNDLED_STATIC}")
