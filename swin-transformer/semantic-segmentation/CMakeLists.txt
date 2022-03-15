cmake_minimum_required(VERSION 3.4)

project(swintransformer)

set(OpenCV_DIR "D:\\opencv\\opencv346\\build")
set(TENSORRT_DIR "D:\\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6\\TensorRT-7.0.0.11")

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
if(WIN32)
include_directories(${PROJECT_SOURCE_DIR}/include)
endif(WIN32)


find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)
enable_language(CUDA)  # 这一句添加后 ，就会在vs中不需要再手动设置cuda 
include_directories(${TENSORRT_DIR}\\include)
link_directories(${TENSORRT_DIR}\\lib)

# file(GLOB SOURCE_FILES "*.cu")
# cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/API.h)
# target_link_libraries(myplugins nvinfer cudart)

# 设置opencv的信息
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB SOURCE_FILES "*.h" "*.cpp" "*.cu")
add_executable(swintransformer ${SOURCE_FILES})

target_link_libraries(swintransformer nvinfer nvonnxparser)
target_link_libraries(swintransformer cudart)
target_link_libraries(swintransformer ${OpenCV_LIBS})

# if (WIN32)
    # message(STATUS "copy dll......: ${CMAKE_COMMAND} ${TENSORRT_DIR}")
    # add_custom_command(TARGET swintransformer POST_BUILD
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DIR}/lib/myelin64_1.dll ./${CMAKE_BUILD_TYPE}/myelin64_1.dll
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DIR}/lib/nvinfer.dll ./${CMAKE_BUILD_TYPE}/nvinfer.dll
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DIR}/lib/nvinfer_plugin.dll ./${CMAKE_BUILD_TYPE}/nvinfer_plugin.dll
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DIR}/lib/nvonnxparser.dll ./${CMAKE_BUILD_TYPE}/nvonnxparser.dll
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DIR}/lib/nvparsers.dll ./${CMAKE_BUILD_TYPE}/nvparsers.dll
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DIR}/lib/nvserialize.dll ./${CMAKE_BUILD_TYPE}/nvserialize.dll
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDA_TOOLKIT_ROOT_DIR}/bin/cublas64_10.dll ./${CMAKE_BUILD_TYPE}/cublas64_10.dll
    # )
# endif(WIN32)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)