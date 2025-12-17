cmake_minimum_required(VERSION 3.10)

project(yolov12)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Set CUDA compiler - use find_package or environment variable
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(
    CMAKE_CUDA_COMPILER nvcc
    HINTS ENV CUDA_HOME
    PATH_SUFFIXES bin)
endif()
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt
# Use CUDA_TOOLKIT_ROOT_DIR or CUDA_HOME environment variable
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
  else()
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
  endif()
endif()

# Use TENSORRT_DIR environment variable or default path
if(NOT DEFINED TENSORRT_DIR)
  if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
  else()
    set(TENSORRT_DIR "/opt/TensorRT-8.6.1.6")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)
  link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib)
else()
  message("embed_platform off")

  # cuda
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  # tensorrt
  include_directories(${TENSORRT_DIR}/include)
  link_directories(${TENSORRT_DIR}/lib)
endif()

add_library(
  myplugins SHARED
  ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(
  GLOB_RECURSE SRCS
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(
  yolov12-det
  ${PROJECT_SOURCE_DIR}/yolov12_det.cpp
  ${SRCS})
target_link_libraries(
  yolov12-det nvinfer cudart myplugins ${OpenCV_LIBS})

add_executable(
  yolov12-seg
  ${PROJECT_SOURCE_DIR}/yolov12_seg.cpp
  ${SRCS})
target_link_libraries(
  yolov12-seg nvinfer cudart myplugins ${OpenCV_LIBS})

add_executable(
  yolov12-cls
  ${PROJECT_SOURCE_DIR}/yolov12_cls.cpp
  ${SRCS})
target_link_libraries(
  yolov12-cls nvinfer cudart myplugins ${OpenCV_LIBS})
