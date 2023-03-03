cmake_minimum_required(VERSION 3.1)

project(rcnn)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--extended-lambda)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/jushi/TensorRT-8.2.1.6/include)
link_directories(/home/jushi/TensorRT-8.2.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/BatchedNms.cu ${PROJECT_SOURCE_DIR}/PredictorDecode.cu ${PROJECT_SOURCE_DIR}/RoiAlign.cu ${PROJECT_SOURCE_DIR}/RpnDecode.cu ${PROJECT_SOURCE_DIR}/RpnNms.cu ${PROJECT_SOURCE_DIR}/MaskRcnnInference.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rcnn ${PROJECT_SOURCE_DIR}/rcnn.cpp)
target_link_libraries(rcnn nvinfer)
target_link_libraries(rcnn cudart)
target_link_libraries(rcnn myplugins)
target_link_libraries(rcnn ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

