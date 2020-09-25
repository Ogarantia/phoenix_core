cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(BUILD_TESTS    "Build tests" ON)
option(WITH_ONEDNN    "Enable oneDNN backend (CPU)" ON)
option(WITH_CUDNN     "Enable cuDNN backend (GPU)" OFF)
option(WITH_FP16      "Enable fp16 support on GPU" ON)
option(WITH_FP16      "Enable fp16 support on GPU" ON)
option(ALLOW_VERBOSE  "Allows verbose messaging" ON)

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- Used backends:")
message("     oneDNN: ${WITH_ONEDNN}")
message("     cuDNN: ${WITH_CUDNN}")
if (WITH_CUDNN)
  message("       fp16 support: ${WITH_FP16}")
endif()

# list source code files
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
set(CPP_SOURCES_FOLDER "${CMAKE_CURRENT_LIST_DIR}/src")
set(SOURCES ${SOURCES}
    ${CPP_SOURCES_FOLDER}/backend/backend.cc
    ${CPP_SOURCES_FOLDER}/algebras.cc
    ${CPP_SOURCES_FOLDER}/utils.cc
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(OPTIMIZATION_FLAGS "-O3 -ffast-math -funroll-loops")
  set(VECTORIZATION_FLAGS "-ftree-vectorize -march=native")
else ()
  set(OPTIMIZATION_FLAGS "-O3 -ffast-math -funroll-loops -floop-nest-optimize ")
  set(VECTORIZATION_FLAGS "-ftree-vectorize -march=native")
endif()

if (ALLOW_VERBOSE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUPSTRIDE_ALLOW_VERBOSE")
  string(ASCII 27 Esc)
  set(ColorCyan  "${Esc}[36m")
  set(ColorReset "${Esc}[m")
  message("-- ${ColorCyan}VERBOSE MESSAGES ENABLED. This is not a production build.${ColorReset}")
endif()

#######################
#    ONEDNN SETUP     #
#######################

if (WITH_ONEDNN)
  # add dependency to oneDNN
  set(ONEDNN_SUBMODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/thirdparty/onednn")
  option(DNNL_BUILD_TESTS "Build DNLL tests" OFF)
  option(DNNL_BUILD_EXAMPLES "Build DNLL examples" OFF)
  set(DNNL_ARCH_OPT_FLAGS "${OPTIMIZATION_FLAGS} ${VECTORIZATION_FLAGS}" CACHE INTERNAL "" )
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
  set(CMAKE_CXX_FLAGS "-DNDEBUG -DBACKEND_CUDNN ${OPTIMIZATION_FLAGS} ${CMAKE_CXX_FLAGS}")
  if (WITH_FP16)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_53,code=sm_53 -gencode arch=compute_75,code=sm_75")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUPSTRIDE_ENABLE_FP16")
  endif()
  set(SOURCES ${SOURCES}
      ${CPP_SOURCES_FOLDER}/backend/cudnn/conv2d_algo_select.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/cudnn.cc
      ${CPP_SOURCES_FOLDER}/backend/cudnn/kernels.cu
      ${CPP_SOURCES_FOLDER}/backend/cudnn/kernels_quat.cu
  )
  link_libraries(cudnn)
endif()

#######################
#        TESTS        #
#######################

if (BUILD_TESTS)
  enable_testing()
  include_directories(tests/cpp_tests/doctest)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/src/cc)

  add_executable(tests
      ${SOURCES}
      ${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp)
  
  add_test(tests tests)
endif()
