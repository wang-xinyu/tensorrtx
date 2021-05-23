cmake_minimum_required(VERSION 3.10)

# set the project name
project(densenet)

add_definitions(-std=c++11)

# get main project dir to include common files
get_filename_component(MAIN_DIR ../ ABSOLUTE)

# When enabled the static version of the 
# CUDA runtime library will be used in CUDA_LIBRARIES
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# include

# include and link cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# include and link tensorrt
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

# add the executable
add_executable(densenet ${PROJECT_SOURCE_DIR}/densenet121.cpp)

target_link_libraries(densenet nvinfer)
target_link_libraries(densenet cudart)

add_definitions(-O2 -pthread)