cmake_minimum_required(VERSION 3.16)
project(real-esrgan)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

#find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/include)

# cuda
FIND_PACKAGE(CUDA REQUIRED)
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# <------------------------TensorRT Related------------------------->
include_directories(YOUR_TENSORRT_INCLUDE_DIR) # TensorRT-8.6.1.6/include
link_directories(YOUR_TENSORRT_LIB_DIR) # TensorRT-8.6.1.6/lib

# <------------------------OpenCV Related------------------------->
# opencv
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/pixel_shuffle/pixel_shuffle.cu)
target_link_libraries(myplugins nvinfer cudart)


TARGET_LINK_LIBRARIES(${PROJECT_NAME} nvinfer)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} cudart)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} myplugins)
