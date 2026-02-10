cmake_minimum_required(VERSION 3.17.0)

function(_guess_path var_name required_files)
  set(_result "")

  foreach(path_entry IN LISTS ARGN)
    if(NOT EXISTS "${path_entry}")
      message(DEBUG "skip non-existing path '${path_entry}'")
      continue()
    endif()

    set(_ok TRUE)
    foreach(required_file IN LISTS required_files)
      if(NOT EXISTS "${path_entry}/${required_file}")
        set(_ok FALSE)
        message(DEBUG "'${path_entry}' missing '${required_file}'")
        break()
      endif()
    endforeach()

    if(_ok)
      list(APPEND _result "${path_entry}")
      message(DEBUG "accept '${path_entry}'")
    else()
      message(DEBUG "reject '${path_entry}'")
    endif()
  endforeach()

  if(_result STREQUAL "")
    message(
      FATAL_ERROR
        "_guess_path(${var_name}) failed: no valid path found. required_files='${required_files}' candidates='${ARGN}'"
    )
  endif()

  set(${var_name}
      "${_result}"
      PARENT_SCOPE)
endfunction()

# add library
add_library(TensorRT IMPORTED INTERFACE)
add_library(TensorRT::TensorRT ALIAS TensorRT)

set(TRT_VERSION
    CACHE
      STRING
      "TensorRT version, e.g. \"8.6.1.6\" or \"8.6.1.6+cuda12.0.1.011\", \"8.6.1.6.Windows10.x86_64.cuda-12.0\" etc"
)

if(NOT TRT_VERSION STREQUAL "" AND NOT $ENV{TRT_VERSION} STREQUAL "")
  message(
    WARNING
      "TRT_VERSION defined by cmake and environment variable both, using the later one"
  )
endif()

if(NOT $ENV{TRT_VERSION} STREQUAL "")
  set(TRT_VERSION $ENV{TRT_VERSION})
endif()

string(REGEX MATCH "([0-9]+)" _match ${TRT_VERSION})
set(TRT_MAJOR_VERSION "${_match}")
unset(_match)

if(WIN32)
  set(TensorRT_DIR "C:/Program Files/TensorRT-${TRT_VERSION}")
  if(NOT EXISTS "${TensorRT_DIR}")
    message(
      FATAL_ERROR
        "TensorRT_DIR=${TensorRT_DIR} does not exist!"
    )
  endif()

  if(${TRT_MAJOR_VERSION} GREATER_EQUAL 10)
    set(_modules nvinfer_10 nvinfer_plugin_10 nvinfer_vc_plugin_10
                 nvinfer_dispatch_10 nvinfer_lean_10)
    message(DEBUG "Using ${_modules}")
  else()
    set(_modules nvinfer nvinfer_plugin nvinfer_vc_plugin nvinfer_dispatch
                 nvinfer_lean)
  endif()

  set(TensorRT_LIBRARY_DIR "${TensorRT_DIR}/lib")
  set(TensorRT_INCLUDE_DIR "${TensorRT_DIR}/include")
elseif(UNIX)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _trt_arch)
  set(_trt_include_candidates)
  if(_trt_arch MATCHES "^(aarch64|arm64|arch64)$")
    set(_trt_include_candidates "/usr/include/aarch64-linux-gnu" "/usr/include"
                                "/usr/local/cuda/targets/aarch64-linux/include")
    set(_trt_library_candidates
        "/usr/local/tensorrt/targets/aarch64-linux-gnu/lib"
        "/usr/lib/aarch64-linux-gnu" "/usr/lib/aarch64-linux-gnu/tegra"
        "/usr/lib")
  elseif(_trt_arch MATCHES "^(x86_64|amd64)$")
    set(_trt_include_candidates
        "/usr/local/tensorrt/targets/x86_64-linux-gnu/include"
        "/usr/include/x86_64-linux-gnu" "/usr/include")
    set(_trt_library_candidates
        "/usr/local/tensorrt/targets/x86_64-linux-gnu/lib"
        "/usr/lib/x86_64-linux-gnu" "/usr/lib")
  else()
    message(FATAL_ERROR "Unknown architecture")
  endif()

  set(_modules nvinfer nvinfer_plugin)
  if(${TRT_MAJOR_VERSION} GREATER_EQUAL 8)
    list(APPEND _modules nvinfer_vc_plugin nvinfer_dispatch nvinfer_lean)
  endif()

  _guess_path(TensorRT_LIBRARY_DIR "libnvinfer.so;libnvinfer_plugin.so"
              ${_trt_library_candidates})
  message(STATUS "TensorRT libraries: ${TensorRT_LIBRARY_DIR}")
  _guess_path(TensorRT_INCLUDE_DIR "NvInfer.h" ${_trt_include_candidates})
  message(STATUS "TensorRT includes: ${TensorRT_INCLUDE_DIR}")
endif()

foreach(lib IN LISTS _modules)
  find_library(
    TensorRT_${lib}_LIBRARY
    NAMES ${lib}
    HINTS ${TensorRT_LIBRARY_DIR})
  list(APPEND TensorRT_LIBRARIES ${TensorRT_${lib}_LIBRARY})
endforeach()

target_link_libraries(TensorRT INTERFACE ${TensorRT_LIBRARIES})

message(STATUS "Found TensorRT libs: ${TensorRT_LIBRARIES}")

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
unset(_trt_include_candidates)
unset(_trt_library_candidates)
unset(_trt_arch)
