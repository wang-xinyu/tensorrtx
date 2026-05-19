cmake_minimum_required(VERSION 3.17.0)

# Auto-locate nvcc when running from IDEs (e.g. CLion) where PATH may not
# include CUDA. Must run BEFORE project() enables the CUDA language.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  find_program(
    _NVCC_EXECUTABLE
    NAMES nvcc
    HINTS ENV CUDA_PATH ENV CUDA_HOME
    PATHS /usr/local/cuda/bin /usr/local/cuda-12/bin /usr/local/cuda-11/bin
          /opt/cuda/bin
    PATH_SUFFIXES bin
    DOC "NVIDIA CUDA compiler")
  if(_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER
        "${_NVCC_EXECUTABLE}"
        CACHE FILEPATH "CUDA compiler" FORCE)
    message(STATUS "Auto-detected CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  else()
    message(
      WARNING
        "nvcc not found via auto-detection. Set -DCMAKE_CUDA_COMPILER=/path/to/nvcc"
        " or env CUDACXX, or add CUDA_PATH/CUDA_HOME to environment.")
  endif()
endif()

project(
  vit
  VERSION 0.1
  LANGUAGES C CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
endif()

set(CMAKE_CXX_STANDARD 20)
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
  include(${CMAKE_SOURCE_DIR}/FindTensorRT.cmake)
else()
  message("TensorRT has been found, skipping for ${PROJECT_NAME}")
endif()

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cc" "cuda_allocator.cc"
                               "profiler.cc")
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  ${PROJECT_NAME} PUBLIC Threads::Threads CUDA::cudart CUDA::cuda_driver
                         TensorRT::TensorRT ${OpenCV_LIBS})

if(WIN32)
  set_target_properties(
    ${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY
                               "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
