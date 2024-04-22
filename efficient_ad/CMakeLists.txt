cmake_minimum_required(VERSION 3.12)
project(EfficientAD-M)

add_definitions(-w)
add_definitions(-D API_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /od")

### nvcc
set(CMAKE_CUDA_COMPILER "D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
enable_language(CUDA)
### cuda
include_directories("D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
link_directories("D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
### tensorrt
set(TRT_DIR "D:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.5.3.1/")
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)
### opencv
set(OpenCV_DIR "E:/OpenCV/OpenCV_4.6.0/opencv/build")
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
### dirent
include_directories("E:/SDK/dirent-1.24/include")

include_directories(${PROJECT_SOURCE_DIR}/src/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(efficientAD_det "./efficientAD_det.cpp" ${SRCS})
target_link_libraries(efficientAD_det nvinfer
                                      cudart
                                      nvinfer_plugin
                                      ${OpenCV_LIBS}
                                      )
