cmake_minimum_required(VERSION 2.6)

project(yolov7)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different


if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    # cuda
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    # tensorrt
    include_directories(/home/na/TensorRT-7.2.2.3/include)
    link_directories(/home/na/TensorRT-7.2.2.3/lib)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(yolov7 calibrator.cpp yolov7.cpp preprocess.cu)

target_link_libraries(yolov7 nvinfer)
target_link_libraries(yolov7 cudart)
target_link_libraries(yolov7 myplugins)
target_link_libraries(yolov7 ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


