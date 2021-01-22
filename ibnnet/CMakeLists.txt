cmake_minimum_required(VERSION 2.6)

project(IBNNet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

file(GLOB SOURCE_FILES "*.h" "*.cpp")

add_executable(ibnnet ${SOURCE_FILES})
target_link_libraries(ibnnet nvinfer)
target_link_libraries(ibnnet cudart)
target_link_libraries(ibnnet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

