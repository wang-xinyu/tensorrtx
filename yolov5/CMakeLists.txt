cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5_det yolov5_det.cpp ${SRCS})
target_link_libraries(yolov5_det nvinfer)
target_link_libraries(yolov5_det cudart)
target_link_libraries(yolov5_det myplugins)
target_link_libraries(yolov5_det ${OpenCV_LIBS})

add_executable(yolov5_cls yolov5_cls.cpp ${SRCS})
target_link_libraries(yolov5_cls nvinfer)
target_link_libraries(yolov5_cls cudart)
target_link_libraries(yolov5_cls myplugins)
target_link_libraries(yolov5_cls ${OpenCV_LIBS})

add_executable(yolov5_seg yolov5_seg.cpp ${SRCS})
target_link_libraries(yolov5_seg nvinfer)
target_link_libraries(yolov5_seg cudart)
target_link_libraries(yolov5_seg myplugins)
target_link_libraries(yolov5_seg ${OpenCV_LIBS})

