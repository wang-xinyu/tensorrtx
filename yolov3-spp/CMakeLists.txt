cmake_minimum_required(VERSION 2.6)

project(yolov3-spp)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)
cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov3-spp ${PROJECT_SOURCE_DIR}/yolov3-spp.cpp)
target_link_libraries(yolov3-spp nvinfer)
target_link_libraries(yolov3-spp cudart)
target_link_libraries(yolov3-spp yololayer)
target_link_libraries(yolov3-spp ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

