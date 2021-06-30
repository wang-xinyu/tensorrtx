cmake_minimum_required(VERSION 2.6)

project(detr)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# tensorrt
include_directories(/home/jushi/TensorRT-7.2.1.6/include)
link_directories(/home/jushi/TensorRT-7.2.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detr ${PROJECT_SOURCE_DIR}/detr.cpp)
target_link_libraries(detr nvinfer)
target_link_libraries(detr cudart)
target_link_libraries(detr ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

