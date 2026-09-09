cmake_minimum_required(VERSION 3.17.0)

# ---------------------------------------------------------------------------
# CUDA is optional: auto-detected, or force off with -DMULTEM_ENABLE_CUDA=OFF
# ---------------------------------------------------------------------------
option(MULTEM_ENABLE_CUDA "Build with CUDA GPU support" ON)

if(MULTEM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    project(multem_tests CXX CUDA)
    set(MULTEM_HAS_CUDA TRUE)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
  else()
    project(multem_tests CXX)
    set(MULTEM_HAS_CUDA FALSE)
    message(STATUS "CUDA not found -- building CPU-only")
  endif()
else()
  project(multem_tests CXX)
  set(MULTEM_HAS_CUDA FALSE)
  message(STATUS "CUDA disabled by user -- building CPU-only")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA architecture (only when CUDA enabled)
if(MULTEM_HAS_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if ($ENV{CMAKE_CUDA_ARCHITECTURES})
      set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
    else()
      set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()
  endif()
endif()

# FFTW finder module (reuse from python-multem)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../python-multem/cmake)

# Dependencies
if(MULTEM_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
find_package(FFTW REQUIRED)

# OpenMP (optional but strongly recommended for CPU path)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
  message(STATUS "OpenMP not found -- CPU path will use std::thread only")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Architecture-specific optimisation flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  # Apple Silicon (M1/M2/M3/M4) or generic AArch64
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    message(STATUS "Apple Silicon detected -- using -mcpu=apple-m1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  message(STATUS "x86_64 detected -- using -march=native")
endif()

# CUDA flags
if(MULTEM_HAS_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
endif()

# Paths to source trees
if(NOT MULTEM_SOURCE_ROOT)
  set(MULTEM_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../python-multem)
endif()
set(PYTHON_MULTEM_SRC ${MULTEM_SOURCE_ROOT}/src)
set(MULTEM_UPSTREAM_SRC ${MULTEM_SOURCE_ROOT}/MULTEM/src)

# Stub include paths for CPU-only builds (thrust/CUDA/CUB replacements)
set(MULTEM_STUBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# ---------------------------------------------------------------------------
# Build MULTEM wrapper as a static library
# ---------------------------------------------------------------------------
if(MULTEM_HAS_CUDA)
  # GPU build: compile .cu with nvcc
  add_library(multem_lib STATIC
    ${PYTHON_MULTEM_SRC}/multem/multem_ext.cu
  )
  set_property(TARGET multem_lib PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

  target_include_directories(multem_lib PUBLIC
    ${PYTHON_MULTEM_SRC}
    ${MULTEM_UPSTREAM_SRC}
    ${FFTW_INCLUDES}
  )

  target_link_libraries(multem_lib PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cufft
    ${FFTW_LIBRARIES}
  )
else()
  # CPU-only build: compile .cu as C++ with stub headers
  # Copy multem_ext.cu to .cpp for the C++ compiler
  configure_file(
    ${PYTHON_MULTEM_SRC}/multem/multem_ext.cu
    ${CMAKE_CURRENT_BINARY_DIR}/multem_ext.cpp
    COPYONLY
  )

  add_library(multem_lib STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/multem_ext.cpp
  )

  target_compile_definitions(multem_lib PUBLIC MULTEM_NO_CUDA)

  # Stub headers must come BEFORE real source so they shadow CUDA/Thrust headers
  target_include_directories(multem_lib PUBLIC
    ${MULTEM_STUBS_DIR}/cuda_stubs
    ${MULTEM_STUBS_DIR}/thrust_cpu
    ${PYTHON_MULTEM_SRC}
    ${MULTEM_UPSTREAM_SRC}
    ${FFTW_INCLUDES}
  )

  target_link_libraries(multem_lib PUBLIC
    ${FFTW_LIBRARIES}
  )
endif()

# OpenMP linkage (both GPU and CPU builds benefit from threaded CPU path)
if(OpenMP_CXX_FOUND)
  target_link_libraries(multem_lib PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(multem_lib PUBLIC MULTEM_USE_OPENMP)
endif()

# ---------------------------------------------------------------------------
# Test harness executable
# ---------------------------------------------------------------------------
if(MULTEM_HAS_CUDA)
  add_executable(test_multem test_multem.cu)
  set_property(TARGET test_multem PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
else()
  # CPU-only: compile test harness as C++
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multem.cu
    ${CMAKE_CURRENT_BINARY_DIR}/test_multem.cpp
    COPYONLY
  )
  add_executable(test_multem ${CMAKE_CURRENT_BINARY_DIR}/test_multem.cpp)
  target_compile_definitions(test_multem PRIVATE MULTEM_NO_CUDA)
  if(NOT MULTEM_HAS_CUDA)
    target_include_directories(test_multem PRIVATE
      ${MULTEM_STUBS_DIR}/cuda_stubs
      ${MULTEM_STUBS_DIR}/thrust_cpu
    )
  endif()
endif()

target_include_directories(test_multem PRIVATE
  ${PYTHON_MULTEM_SRC}
  ${MULTEM_UPSTREAM_SRC}
)

target_link_libraries(test_multem PRIVATE
  multem_lib
)

# On Apple, link to pthreads explicitly
if(APPLE)
  target_link_libraries(multem_lib PUBLIC pthread)
endif()
