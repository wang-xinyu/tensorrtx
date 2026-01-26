cmake_minimum_required(VERSION 3.10)

project(yolov13)

# Set up environment-based paths for CUDA and TensorRT
if(DEFINED ENV{CUDA_HOME})
  set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
else()
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()

if(DEFINED ENV{TENSORRT_DIR})
  set(TENSORRT_ROOT $ENV{TENSORRT_DIR})
else()
  set(TENSORRT_ROOT "/opt/TensorRT-10.10.0.31")
endif()

message(STATUS "Using CUDA from: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Using TensorRT from: ${TENSORRT_ROOT}")

# Generate compile_commands.json for better IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# CUDA and TensorRT configuration
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib)
  include_directories(${TENSORRT_ROOT}/include)
  link_directories(${TENSORRT_ROOT}/lib)
else()
  message("embed_platform off")
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  include_directories(${TENSORRT_ROOT}/include)
  link_directories(${TENSORRT_ROOT}/lib)
endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(yolov13-det ${PROJECT_SOURCE_DIR}/yolov13_det.cpp ${SRCS})

# Use target-based include for better IDE indexing
target_include_directories(yolov13-det PRIVATE ${TENSORRT_ROOT}/include ${CUDA_TOOLKIT_ROOT_DIR}/include)

target_link_libraries(yolov13-det nvinfer)
target_link_libraries(yolov13-det cudart)
target_link_libraries(yolov13-det myplugins)
target_link_libraries(yolov13-det ${OpenCV_LIBS})
