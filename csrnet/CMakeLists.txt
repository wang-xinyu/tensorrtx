cmake_minimum_required(VERSION 3.10)

project(csrnet)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# cuda
include_directories(/usr/local/cuda/targets/x86_64-linux/include )
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/)

add_executable(csrnet csrnet.cpp)
target_link_libraries(csrnet nvinfer cudart ${OpenCV_LIBS})