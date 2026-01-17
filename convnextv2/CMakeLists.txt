cmake_minimum_required(VERSION 3.10)
project(convnextv2)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/include /usr/local/TensorRT-8.6.1.6/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/local/cuda/lib64 /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib)

# TRT
find_library(NVINFER nvinfer PATHS /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN nvinfer_plugin PATHS /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib NO_DEFAULT_PATH)
find_library(NVPARSERS nvparsers PATHS /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib NO_DEFAULT_PATH)

set(CMAKE_CXX_STANDARD 14)

cuda_add_executable(convnextv2 src/convnextv2.cpp src/LayerNormPlugin.cu)
target_link_libraries(convnextv2 ${NVINFER} ${NVINFER_PLUGIN} ${CUDA_LIBRARIES} ${OpenCV_LIBS})

cuda_add_library(layernorm_plugin SHARED src/LayerNormPlugin.cu)
target_link_libraries(layernorm_plugin ${NVINFER} ${NVINFER_PLUGIN} ${CUDA_LIBRARIES})

# Inference executable
cuda_add_executable(inference_cpp src/inference_cpp.cpp src/LayerNormPlugin.cu)
target_link_libraries(inference_cpp ${NVINFER} ${NVINFER_PLUGIN} ${CUDA_LIBRARIES} ${OpenCV_LIBS})

