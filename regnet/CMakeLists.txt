cmake_minimum_required(VERSION 2.6)

project(regnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

add_executable(regnet ${PROJECT_SOURCE_DIR}/regnet.cpp)
target_link_libraries(regnet nvinfer)
target_link_libraries(regnet cudart)

add_definitions(-O2 -pthread)
