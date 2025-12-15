cmake_minimum_required(VERSION 3.10)

project(yolov12)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  # For aarch64 platforms, use standard CUDA paths
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)
  # tensorrt - Using environment variable or standard paths
  # Users should set TRT_PATH environment variable or adjust accordingly
  if(DEFINED ENV{TRT_PATH})
    include_directories($ENV{TRT_PATH}/include)
    link_directories($ENV{TRT_PATH}/lib)
  else()
    # Default TensorRT installation paths - users should modify according to their setup
    include_directories(/usr/local/tensorrt/include)
    link_directories(/usr/local/tensorrt/lib)
  endif()
endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(yolov12-det ${PROJECT_SOURCE_DIR}/yolov12_det.cpp ${SRCS} )
target_link_libraries(yolov12-det nvinfer cudart myplugins ${OpenCV_LIBS})

add_executable(yolov12-seg ${PROJECT_SOURCE_DIR}/yolov12_seg.cpp ${SRCS})
target_link_libraries(yolov12-seg nvinfer cudart myplugins ${OpenCV_LIBS})

add_executable(yolov12-cls ${PROJECT_SOURCE_DIR}/yolov12_cls.cpp ${SRCS})
target_link_libraries(yolov12-cls nvinfer cudart myplugins ${OpenCV_LIBS})
