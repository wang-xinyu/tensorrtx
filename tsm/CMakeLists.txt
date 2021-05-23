cmake_minimum_required(VERSION 2.6)

project(TSM)

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
include_directories(/home/ubuntu/TensorRT/include/)
link_directories(/home/ubuntu/TensorRT/lib/)

add_executable(tsm_r50 ${PROJECT_SOURCE_DIR}/tsm_r50.cpp)
target_link_libraries(tsm_r50 nvinfer)
target_link_libraries(tsm_r50 cudart)

add_definitions(-O2 -pthread)
