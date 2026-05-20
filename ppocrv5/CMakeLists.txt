cmake_minimum_required(VERSION 3.10)

project(ppocrv5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(TENSORRT_ROOT /home/algo/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0/TensorRT-10.13.3.9 CACHE PATH
    "TensorRT root")
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include CACHE PATH "TensorRT include dir")
if (NOT TENSORRT_LIB_DIR)
  if (EXISTS "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib CACHE PATH "TensorRT library dir")
  else()
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib CACHE PATH "TensorRT library dir")
  endif()
endif()
set(CUDA_INCLUDE_DIR /usr/local/cuda/include CACHE PATH "CUDA include dir")
set(CUDA_LIB_DIR /usr/local/cuda/lib64 CACHE PATH "CUDA library dir")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "CUDA compiler")
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")

  # cuda
  include_directories(${CUDA_INCLUDE_DIR})
  link_directories(${CUDA_LIB_DIR})

  # tensorrt
  include_directories(${TENSORRT_INCLUDE_DIR})
  link_directories(${TENSORRT_LIB_DIR})
  set(CMAKE_BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
endif()

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(ppocrv5_core STATIC ${SRCS})
target_link_libraries(ppocrv5_core nvinfer)
target_link_libraries(ppocrv5_core cudart)
target_link_libraries(ppocrv5_core ${OpenCV_LIBS})

add_executable(ppocrv5_det ${PROJECT_SOURCE_DIR}/ppocrv5_det.cpp)
set_target_properties(ppocrv5_det PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_det ppocrv5_core)

add_executable(ppocrv5_rec ${PROJECT_SOURCE_DIR}/ppocrv5_rec.cpp)
set_target_properties(ppocrv5_rec PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_rec ppocrv5_core)

add_executable(ppocrv5_ocr ${PROJECT_SOURCE_DIR}/ppocrv5_ocr.cpp)
set_target_properties(ppocrv5_ocr PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_ocr ppocrv5_core)

add_executable(ppocrv5_model ${PROJECT_SOURCE_DIR}/ppocrv5_model.cpp)
set_target_properties(ppocrv5_model PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_model ppocrv5_core)

add_executable(ppocrv5_dump ${PROJECT_SOURCE_DIR}/ppocrv5_dump.cpp)
set_target_properties(ppocrv5_dump PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_dump ppocrv5_core)

add_executable(ppocrv5_formula ${PROJECT_SOURCE_DIR}/ppocrv5_formula.cpp)
set_target_properties(ppocrv5_formula PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_formula ppocrv5_core)

add_executable(ppocrv5_system ${PROJECT_SOURCE_DIR}/ppocr_system.cpp)
set_target_properties(ppocrv5_system PROPERTIES OUTPUT_NAME ppocr_system)
set_target_properties(ppocrv5_system PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}")
target_link_libraries(ppocrv5_system ppocrv5_core)
