cmake_minimum_required(VERSION 2.6)

project(resnet)

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
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

add_executable(resnet18 ${PROJECT_SOURCE_DIR}/resnet18.cpp)
target_link_libraries(resnet18 nvinfer)
target_link_libraries(resnet18 cudart)

add_executable(resnet50 ${PROJECT_SOURCE_DIR}/resnet50.cpp)
target_link_libraries(resnet50 nvinfer)
target_link_libraries(resnet50 cudart)

add_executable(resnext50 ${PROJECT_SOURCE_DIR}/resnext50_32x4d.cpp)
target_link_libraries(resnext50 nvinfer)
target_link_libraries(resnext50 cudart)

add_executable(wideresnet50 ${PROJECT_SOURCE_DIR}/wideresnet50.cpp)
target_link_libraries(wideresnet50 nvinfer)
target_link_libraries(wideresnet50 cudart)

add_definitions(-O2 -pthread)

