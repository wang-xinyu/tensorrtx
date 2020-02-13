cmake_minimum_required(VERSION 2.6)

project(mobilenetv3)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30;)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/targets/aarch64-linux/include)
link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
cuda_add_library(h_sigmoid ${PROJECT_SOURCE_DIR}/h_sigmoid.cu)
#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)

add_executable(mobilenetv3  ${PROJECT_SOURCE_DIR}/h_sigmoidplugin.cpp ${PROJECT_SOURCE_DIR}/mobilenet_v3.cpp)
target_link_libraries(mobilenetv3 nvinfer)
target_link_libraries(mobilenetv3 cudart)
#target_link_libraries(mobilenetv3 leaky)
target_link_libraries(mobilenetv3 h_sigmoid)

add_definitions(-O2 -pthread)

