cmake_minimum_required(VERSION 2.6)

project(LPRnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    # tensorrt
    include_directories(/usr/local/TensorRT-7.0.0.11/include)
    link_directories(/usr/local/TensorRT-7.0.0.11/lib)
endif()

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(LPRnet ${PROJECT_SOURCE_DIR}/LPRnet.cpp)
target_link_libraries(LPRnet nvinfer)
target_link_libraries(LPRnet cudart)
target_link_libraries(LPRnet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)