cmake_minimum_required(VERSION 2.6)

project(mobilenetv3)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode arch=compute_30,code=sm_30;)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

add_executable(mobilenetv3  ${PROJECT_SOURCE_DIR}/mobilenet_v3.cpp)
target_link_libraries(mobilenetv3 nvinfer)
target_link_libraries(mobilenetv3 cudart)

add_definitions(-O2 -pthread)

