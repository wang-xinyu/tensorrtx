cmake_minimum_required(VERSION 2.6)

project(unet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

# cuda directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-10.2/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-10.2/targets/x86_64-linux/lib)

# tensorrt
include_directories(/home/sycv/workplace/pengyuzhou/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/include)
link_directories(/home/sycv/workplace/pengyuzhou/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# link library and add exec file
add_executable(unet ${PROJECT_SOURCE_DIR}/unet.cpp)
target_link_libraries(unet nvinfer)
target_link_libraries(unet cudart)

add_definitions(-O2 -pthread)

# opencv library
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(unet ${OpenCV_LIBS})
