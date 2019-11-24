cmake_minimum_required(VERSION 2.6)

project(googlenet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-9.0/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-9.0/targets/aarch64-linux/lib)

add_executable(googlenet ${PROJECT_SOURCE_DIR}/googlenet.cpp)
target_link_libraries(googlenet nvinfer)
target_link_libraries(googlenet cudart)

add_definitions(-O2 -pthread)

