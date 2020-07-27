cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(BUILD_TESTS "Build tests" ON)
option(WITH_CUDA "With CUDA" OFF)
option(WITH_ONEDNN "With oneDNN" ON)

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- WITH_ONEDNN: ${WITH_ONEDNN}")
message("-- WITH_CUDA: ${WITH_CUDA}")

# CUDA-related stuff
if (WITH_CUDA)
  set(CMAKE_CXX_FLAGS "-DNDEBUG ${CMAKE_CXX_FLAGS}")
  find_package(CUDA)
  set(EXTENSION "cu.cc")
else()
  set(EXTENSION "cc")
endif()

# list source code files
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
set(CPP_SOURCES_FOLDER "${CMAKE_CURRENT_LIST_DIR}/src")
set(SOURCES ${SOURCES} 
    ${CPP_SOURCES_FOLDER}/utils.cc
)
if (WITH_ONEDNN)
  set(SOURCES ${SOURCES}
      ${CPP_SOURCES_FOLDER}/onednn/conv2d.cc
  )
endif()

#######################
#    ONEDNN SETUP     #
#######################

# bring up oneDNN
if (WITH_ONEDNN)
  set(ONEDNN_SUBMODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/thirdparty/onednn")
  option(DNNL_BUILD_TESTS "Build DNLL tests" OFF)
  option(DNNL_BUILD_EXAMPLES "Build DNLL examples" OFF)
  set(DNNL_ARCH_OPT_FLAGS "-O3 -ffast-math -funroll-loops -floop-nest-optimize -ftree-vectorize -msse4.1" CACHE INTERNAL "" )
  add_subdirectory(${ONEDNN_SUBMODULE_PATH})
  include_directories("${ONEDNN_SUBMODULE_PATH}/include")
  link_libraries(dnnl)
endif()


if (BUILD_TESTS)
  enable_testing()
  include_directories(tests/cpp_tests/doctest)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/src/cc)

  set(TESTSOURCES ${TESTSOURCES} 
    ${CPP_SOURCES_FOLDER}/utils.cc
    ${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp
  )

  add_executable(tests ${TESTSOURCES})
  add_test(tests tests)
endif()
