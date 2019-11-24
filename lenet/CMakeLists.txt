cmake_minimum_required(VERSION 2.6)

project(lenet)

add_definitions(-std=c++11)

set(TARGET_NAME "lenet")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-9.0/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-9.0/targets/aarch64-linux/lib)

FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} nvinfer)
target_link_libraries(${TARGET_NAME} cudart)

add_definitions(-O2 -pthread)

