cmake_minimum_required(VERSION 3.17.0)

project(
  lenet
  VERSION 0.1
  LANGUAGES C CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      75
      80
      86
      89
      90
      100
      120)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

option(CUDA_USE_STATIC_CUDA_RUNTIME "Use static cudaruntime library" OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT TARGET TensorRT::TensorRT)
  include(FindTensorRT.cmake)
else()
  message("TensorRT has been found, skipping for ${PROJECT_NAME}")
endif()

add_executable(${PROJECT_NAME} lenet.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                                                   ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads CUDA::cudart
                                             TensorRT::TensorRT ${OpenCV_LIBS})

if(WIN32)
  set_target_properties(
    ${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY
                               "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
