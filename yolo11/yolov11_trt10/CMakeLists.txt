cmake_minimum_required(VERSION 3.10)

project(yolov11)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
add_definitions(-DNOMINMAX)          # 新增：防止windows.h定义min/max宏
add_definitions(-DWIN32_LEAN_AND_MEAN) # 新增：减少windows.h包含的内容
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")

  # cuda
  include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
  link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib")

  # tensorrt
  include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.13.0.35_cuda-11.8/include")
  link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.13.0.35_cuda-11.8/lib")
endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer_10 cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(yolo11_det ${PROJECT_SOURCE_DIR}/yolo11_det.cpp ${SRCS})
target_link_libraries(yolo11_det nvinfer_10)
target_link_libraries(yolo11_det cudart)
target_link_libraries(yolo11_det myplugins)
target_link_libraries(yolo11_det ${OpenCV_LIBS})

add_executable(yolo11_cls ${PROJECT_SOURCE_DIR}/yolo11_cls.cpp ${SRCS})
target_link_libraries(yolo11_cls nvinfer_10)
target_link_libraries(yolo11_cls cudart)
target_link_libraries(yolo11_cls myplugins)
target_link_libraries(yolo11_cls ${OpenCV_LIBS})

add_executable(yolo11_seg ${PROJECT_SOURCE_DIR}/yolo11_seg.cpp ${SRCS})
target_link_libraries(yolo11_seg nvinfer_10)
target_link_libraries(yolo11_seg cudart)
target_link_libraries(yolo11_seg myplugins)
target_link_libraries(yolo11_seg ${OpenCV_LIBS})

add_executable(yolo11_pose ${PROJECT_SOURCE_DIR}/yolo11_pose.cpp ${SRCS})
target_link_libraries(yolo11_pose nvinfer_10)
target_link_libraries(yolo11_pose cudart)
target_link_libraries(yolo11_pose myplugins)
target_link_libraries(yolo11_pose ${OpenCV_LIBS})

add_executable(yolo11_obb ${PROJECT_SOURCE_DIR}/yolo11_obb.cpp ${SRCS})
target_link_libraries(yolo11_obb nvinfer_10)
target_link_libraries(yolo11_obb cudart)
target_link_libraries(yolo11_obb myplugins)
target_link_libraries(yolo11_obb ${OpenCV_LIBS})
