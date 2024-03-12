cmake_minimum_required(VERSION 3.10)

project(yolov8)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  include_directories(/home/lindsay/TensorRT-8.4.1.5/include)
  link_directories(/home/lindsay/TensorRT-8.4.1.5/lib)
#  include_directories(/home/lindsay/TensorRT-7.2.3.4/include)
#  link_directories(/home/lindsay/TensorRT-7.2.3.4/lib)


endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8_det ${PROJECT_SOURCE_DIR}/yolov8_det.cpp ${SRCS})

target_link_libraries(yolov8_det nvinfer)
target_link_libraries(yolov8_det cudart)
target_link_libraries(yolov8_det myplugins)
target_link_libraries(yolov8_det ${OpenCV_LIBS})

add_executable(yolov8_seg ${PROJECT_SOURCE_DIR}/yolov8_seg.cpp ${SRCS})
target_link_libraries(yolov8_seg nvinfer cudart myplugins ${OpenCV_LIBS})

add_executable(yolov8_cls ${PROJECT_SOURCE_DIR}/yolov8_cls.cpp ${SRCS})
target_link_libraries(yolov8_cls nvinfer cudart myplugins ${OpenCV_LIBS})