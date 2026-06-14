cmake_minimum_required(VERSION 3.17.0)

set(TRT_VERSION
    $ENV{TRT_VERSION}
    CACHE STRING
          "TensorRT version, e.g. \"8.6.1.6\" or \"8.6.1.6+cuda12.0.1.011\"")

function(_guess_path var_name)
  set(_result "")

  foreach(path_entry IN LISTS ARGN)
    if(EXISTS "${path_entry}")
      list(APPEND _result "${path_entry}")
    endif()
  endforeach()

  set(${var_name}
      "${_result}"
      PARENT_SCOPE)
endfunction()

# find TensorRT include folder
if(NOT DEFINED TensorRT_INCLUDE_DIR)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    _guess_path(TensorRT_INCLUDE_DIR
                "/usr/local/cuda/targets/aarch64-linux/include")
  else()
    _guess_path(
      TensorRT_INCLUDE_DIR
      "/usr/local/tensorrt/targets/x86_64-linux-gnu/include"
      "/usr/include/x86_64-linux-gnu")
  endif()
  message(STATUS "TensorRT includes: ${TensorRT_INCLUDE_DIR}")
endif()

# find TensorRT library folder
if(NOT TensorRT_LIBRARY_DIR)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    _guess_path(TensorRT_LIBRARY_DIR "/usr/lib/aarch64-linux-gnu/tegra")
  else()
    _guess_path(TensorRT_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu"
                "/usr/local/tensorrt/targets/x86_64-linux-gnu/lib")
  endif()
  message(STATUS "TensorRT libraries: ${TensorRT_LIBRARY_DIR}")
endif()

set(TensorRT_LIBRARIES)

message(STATUS "Found TensorRT lib: ${TensorRT_LIBRARIES}")

# process for different TensorRT version
if(DEFINED TRT_VERSION AND NOT TRT_VERSION STREQUAL "")
  string(REGEX MATCH "([0-9]+)" _match ${TRT_VERSION})
  set(TRT_MAJOR_VERSION "${_match}")
  set(_modules nvinfer nvinfer_plugin)
  unset(_match)

  if(TRT_MAJOR_VERSION GREATER_EQUAL 8)
    list(APPEND _modules nvinfer_vc_plugin nvinfer_dispatch nvinfer_lean)
  endif()
else()
  message(FATAL_ERROR "Please set a environment variable \"TRT_VERSION\"")
endif()

# find and add all modules of TensorRT into list
foreach(lib IN LISTS _modules)
  find_library(
    TensorRT_${lib}_LIBRARY
    NAMES ${lib}
    HINTS ${TensorRT_LIBRARY_DIR})
  list(APPEND TensorRT_LIBRARIES ${TensorRT_${lib}_LIBRARY})
endforeach()

# make the "TensorRT target"
add_library(TensorRT IMPORTED INTERFACE)
add_library(TensorRT::TensorRT ALIAS TensorRT)
target_link_libraries(TensorRT INTERFACE ${TensorRT_LIBRARIES})

set_target_properties(
  TensorRT
  PROPERTIES C_STANDARD 17
             CXX_STANDARD 17
             POSITION_INDEPENDENT_CODE ON
             SKIP_BUILD_RPATH TRUE
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
             INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")

unset(TRT_MAJOR_VERSION)
unset(_modules)
