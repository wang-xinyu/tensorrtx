cmake_minimum_required(VERSION 2.6)

project(senet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-9.0/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-9.0/targets/aarch64-linux/lib)

add_executable(se_resnet ${PROJECT_SOURCE_DIR}/se_resnet50.cpp)
target_link_libraries(se_resnet nvinfer)
target_link_libraries(se_resnet cudart)

add_definitions(-O2 -pthread)

