cmake_minimum_required(VERSION 3.14)

project(
  tensorrtx
  VERSION 0.1
  LANGUAGES C CXX CUDA)

set(TensorRT_7_8_10_TARGETS mlp lenet alexnet vgg)

set(TensorRT_8_TARGETS)

set(TensorRT_10_TARGETS)

set(ALL_TARGETS ${TensorRT_7_8_10_TARGETS} ${TensorRT_8_TARGETS}
                ${TensorRT_10_TARGETS})

foreach(sub_dir ${ALL_TARGETS})
  message(STATUS "Add subdirectory: ${sub_dir}")
  add_subdirectory(${sub_dir})
endforeach()
