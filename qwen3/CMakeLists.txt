cmake_minimum_required(VERSION 3.2)
project(qwen3_trtx LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

find_package(OpenCV REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(TENSORRT_ROOT /workspace/shared/TensorRT-10.10.0.31)
set(TOKENIZER_LIB_DIR /workspace/shared/code/tokenizer.cpp-linux-x86_64)

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_FILES)
file(GLOB PLUGIN_FILES
  ${CMAKE_SOURCE_DIR}/plugin/*.cpp
  ${CMAKE_SOURCE_DIR}/plugin/*.cu)

add_executable(qwen3_trtx qwen3.cpp ${SRC_FILES} ${PLUGIN_FILES})
set_source_files_properties(qwen3.cpp PROPERTIES LANGUAGE CXX)

foreach(src_file IN LISTS SRC_FILES)
  set_source_files_properties(${src_file} PROPERTIES LANGUAGE CXX)
endforeach()

target_include_directories(qwen3_trtx PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${TENSORRT_ROOT}/include
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/plugin
  ${CMAKE_SOURCE_DIR}/src
  ${TOKENIZER_LIB_DIR}/include)

target_compile_definitions(qwen3_trtx PRIVATE
  NOMINMAX
  API_EXPORTS
  UTF8PROC_EXPORTS)

target_link_directories(qwen3_trtx PRIVATE
  ${OpenCV_LIB_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${TENSORRT_ROOT}/lib
  ${CMAKE_SOURCE_DIR}/lib
  ${TOKENIZER_LIB_DIR}/lib)

target_link_libraries(qwen3_trtx PRIVATE
  tokenizer_lib
  opencv_core
  opencv_highgui
  opencv_imgproc
  opencv_imgcodecs
  cudart
  cublas
  nvinfer
  onig)
