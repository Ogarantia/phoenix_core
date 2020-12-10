cmake_minimum_required(VERSION 3.18)

project(UpStride)

option(BUILD_TESTS    "Build tests" ON)
option(WITH_ONEDNN    "Enable oneDNN backend (CPU)" ON)
option(WITH_CUDNN     "Enable cuDNN backend (GPU)" OFF)
option(WITH_FP16      "Enable fp16 support on GPU" ON)
option(UPSTRIDE_DEBUG "Enable development features" ON)
set(ARCH "native" CACHE STRING "Enable optimization for specific arch (x86_64 / aarch64 / <spe_march>)")

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- Arch: ${ARCH}")
message("-- Used backends:")
message("     oneDNN: ${WITH_ONEDNN}")
message("     cuDNN: ${WITH_CUDNN}")

if (WITH_CUDNN)
  message("       fp16 support: ${WITH_FP16}")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# list source code files
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
set(CPP_SOURCES_FOLDER "${CMAKE_CURRENT_LIST_DIR}/src")
set(SOURCES ${SOURCES}
    ${CPP_SOURCES_FOLDER}/backend/backend.cc
    ${CPP_SOURCES_FOLDER}/algebras.cc
    ${CPP_SOURCES_FOLDER}/isolated_thread.cc
    ${CPP_SOURCES_FOLDER}/utils.cc
)

# Common set of flags independent of development or production build
set(COMMON_FLAGS "--std=c++14")
# Set of optimization flags
set(OPTIMIZATION_FLAGS "-O3")
set(VECTORIZATION_FLAGS "-ftree-vectorize")
set(COMPUTE_OPTIMIZATION_FLAGS "-ffast-math -funroll-loops")
# Set of compiler specific compilation flags
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(COMPUTE_OPTIMIZATION_FLAGS "${COMPUTE_OPTIMIZATION_FLAGS} -floop-nest-optimize ")
endif()

# Set of target architecture specific compilation flags
if (ARCH STREQUAL "x86_64")
  set(ARCH_OPTIMIZATION_FLAGS "-march=skylake -mfma -mavx2")
# see https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/AArch64-Options.html for more specific option
elseif (ARCH STREQUAL "aarch64")
  set(ARCH_OPTIMIZATION_FLAGS "-march=armv8-a+crypto+simd -mcpu=cortex-a57+crypto+simd")
# The user can provide a specific arch for what he want to cross compile; it can also be "native"
else()
  set(ARCH_OPTIMIZATION_FLAGS "-march=${ARCH}")
endif()


# Adpat flags for development and debuging
if (UPSTRIDE_DEBUG)
  # Disable all optimization flags when debug, make things easier when using gdb
  set(OPTIMIZATION_FLAGS "-O0 -g -DUPSTRIDE_DEBUG")
  set(VECTORIZATION_FLAGS "")
  set(COMPUTE_OPTIMIZATION_FLAGS "")
  set(ARCH_OPTIMIZATION_FLAGS "")
  string(ASCII 27 Esc)
  set(ColorCyan  "${Esc}[36m")
  set(ColorReset "${Esc}[m")
  message("-- ${ColorCyan}This is NOT a production build.${ColorReset}")
endif()

#######################
#    ONEDNN SETUP     #
#######################

if (WITH_ONEDNN)
  # add dependency to oneDNN
  set(ONEDNN_SUBMODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/thirdparty/onednn")
  option(DNNL_BUILD_TESTS "Build DNNL tests" OFF)
  option(DNNL_BUILD_EXAMPLES "Build DNNL examples" OFF)
  set(DNNL_ARCH_OPT_FLAGS "${COMMON_FLAGS} ${OPTIMIZATION_FLAGS} ${COMPUTE_OPTIMIZATION_FLAGS} ${VECTORIZATION_FLAGS} ${ARCH_OPTIMIZATION_FLAGS}" CACHE INTERNAL "")
  message("**************** Compiling oneDNN: ****************")
  add_subdirectory(${ONEDNN_SUBMODULE_PATH})
  message("**************** ***************** ****************")
  include_directories("${ONEDNN_SUBMODULE_PATH}/include")
  link_libraries(dnnl)

  # add oneDNN-specific sources
  set(SOURCES ${SOURCES}
      ${CPP_SOURCES_FOLDER}/backend/onednn/onednn.cc
      ${CPP_SOURCES_FOLDER}/backend/onednn/tensor.cc
  )
endif()

#######################
#     CUDNN SETUP     #
#######################

if (WITH_CUDNN)
  # set flags
  set(CMAKE_CXX_FLAGS "-DBACKEND_CUDNN ${COMMON_FLAGS} ${OPTIMIZATION_FLAGS} ${COMPUTE_OPTIMIZATION_FLAGS} ${VECTORIZATION_FLAGS} ${ARCH_OPTIMIZATION_FLAGS}")
  set(CMAKE_CUDA_FLAGS "-DBACKEND_CUDNN ${COMMON_FLAGS} ${OPTIMIZATION_FLAGS}")
  if (UPSTRIDE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
    # -g/G generate debug information for respectively host/device https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-device-debug
    # -G cannot be used because it uses more register than we have. However, --generate-line-info is part of -G without playing with register.
    # If you really want to enable -G it's own to you; just put it here.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
  endif()

  # define target GPU architectures the binaries are compiled for
  set(CMAKE_CUDA_ARCHITECTURES 53 60 70 75)
  if (WITH_FP16)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUPSTRIDE_ENABLE_FP16")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUPSTRIDE_ENABLE_FP16")
  else()
    set(CMAKE_CUDA_ARCHITECTURES 37 ${CMAKE_CUDA_ARCHITECTURES})
  endif()


  # list sources
  set(SOURCES ${SOURCES}
      ${CPP_SOURCES_FOLDER}/backend/cudnn/conv2d_algo_select.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/cudnn.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/kernels_utils.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/quat_pointwise_conv2d.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/quat_pointwise_conv2d.cu
      ${CPP_SOURCES_FOLDER}/backend/cudnn/kernels.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/kernels.cu
      ${CPP_SOURCES_FOLDER}/backend/cudnn/kernels_quat.cu
  )

  # specify include path and link
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  link_libraries(cudnn cublas)

endif()

#######################
#        TESTS        #
#######################

if (BUILD_TESTS)
  enable_testing()
  include_directories(tests/cpp_tests/doctest)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/src/cc)

  add_executable(tests ${SOURCES} ${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp)
  add_test(tests tests)
endif()

#######################
#     Misc stuff      #
#######################
if (WITH_CUDNN)
# define a custom target executing a shell script generating a header file that hides function names
set(HIDENAMES_HEADER ${CMAKE_CURRENT_LIST_DIR}/src/backend/cudnn/hidenames.h)
add_custom_target(hidenames ALL
  COMMAND ${CMAKE_CURRENT_LIST_DIR}/scripts/hidenames.sh > ${HIDENAMES_HEADER}.new
  COMMAND  /bin/sh -c "if cmp ${HIDENAMES_HEADER}.new ${HIDENAMES_HEADER} \; \
                       then \
                         rm ${HIDENAMES_HEADER}.new  \; \
                       else \
                         mv ${HIDENAMES_HEADER}.new ${HIDENAMES_HEADER} \; \
                       fi" > /dev/null
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  BYPRODUCTS ${HIDENAMES_HEADER}
  COMMENT "Generating hidenames.h")
endif()