cmake_minimum_required(VERSION 2.6)

project(dbnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIRSRCS)

# clipper
include_directories(./ ./clipper)
add_subdirectory(clipper)

add_executable(dbnet ${DIRSRCS})
target_link_libraries(dbnet clipper)
target_link_libraries(dbnet nvinfer)
target_link_libraries(dbnet cudart)
target_link_libraries(dbnet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

