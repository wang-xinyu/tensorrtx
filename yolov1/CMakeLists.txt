cmake_minimum_required(VERSION 2.6)

project(yolov1)

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
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)
# opencv
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(yolov1 ${PROJECT_SOURCE_DIR}/yolov1.cpp)
target_link_libraries(yolov1 nvinfer)
target_link_libraries(yolov1 cudart)
target_link_libraries(yolov1 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
