cmake_minimum_required(VERSION 2.6)

project(lane_det)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# cuda directory
include_directories(/usr/local/cuda/include/)
link_directories(/usr/local/cuda/lib64/)

# tensorrt
#include_directories(/workspace/TensorRT-7.2.3.4/include/)
#link_directories(/workspace/TensorRT-7.2.3.4/lib/)


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lane_det ${PROJECT_SOURCE_DIR}/lane_det.cpp)
target_link_libraries(lane_det nvinfer)
target_link_libraries(lane_det cudart)
target_link_libraries(lane_det ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

