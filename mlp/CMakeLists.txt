cmake_minimum_required(VERSION 3.14)  # change the version, if asked by compiler
project(mlp)

set(CMAKE_CXX_STANDARD 14)

# include and link dirs of tensorrt, you need adapt them if yours are different
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# create link for executable files
add_executable(mlp mlp.cpp)

# perform linking with nvinfer libraries
target_link_libraries(mlp nvinfer)

# link with cuda libraries for Inference
target_link_libraries(mlp cudart)

add_definitions(-O2 -pthread)

