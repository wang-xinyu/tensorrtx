cmake_minimum_required(VERSION 2.6)

project(refinedet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# tensorrt
include_directories(/data_2/tensorrt/TensorRT-7.0.0.11/include/) #include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/data_2/tensorrt/TensorRT-7.0.0.11/lib/) #link_directories(/usr/lib/x86_64-linux-gnu/)


find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#find_package(OpenCV)
#include_directories(OpenCV_INCLUDE_DIRS)

include_directories(/home/software_install/opencv3.4.6/include)
link_directories(/home/software_install/opencv3.4.6/lib)


set(CMAKE_PREFIX_PATH "/data_1/torch1.1.0") ###torch1.1.0
find_package(Torch REQUIRED)

include_directories(/data_1/torch1.1.0/include)
link_directories(/data_1/torch1.1.0/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


add_executable(refinedet ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/refinedet.cpp)
target_link_libraries(refinedet nvinfer)
target_link_libraries(refinedet cudart)
target_link_libraries(refinedet "${TORCH_LIBRARIES}")
target_link_libraries(refinedet opencv_calib3d opencv_core opencv_dnn opencv_imgproc opencv_highgui opencv_imgcodecs caffe2)

add_definitions(-O2 -pthread)

