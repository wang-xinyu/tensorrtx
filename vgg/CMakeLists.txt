cmake_minimum_required(VERSION 3.14)

project(
  vgg11
  VERSION 0.1
  LANGUAGES C CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      60
      70
      72
      75
      80
      86
      89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
set(CMAKE_BUILD_TYPE
    "Debug"
    CACHE STRING "Build type for this project" FORCE)

option(CUDA_USE_STATIC_CUDA_RUNTIME "Use static cudaruntime library" OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT TARGET TensorRT::TensorRT)
  include(FindTensorRT.cmake)
endif()

add_executable(${PROJECT_NAME} vgg11.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads CUDA::cudart
                                             TensorRT::TensorRT)
