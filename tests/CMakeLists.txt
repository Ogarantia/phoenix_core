include_directories(doctest)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")

add_executable(tests tests.cpp)
add_test(tests tests)

target_link_libraries(tests)