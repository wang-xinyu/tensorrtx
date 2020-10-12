# Frequently Asked Questions (FAQ)

## 1. fatal error: NvInfer.h: No such file or directory

`NvInfer.h` is one of the headers of TensorRT. If you install the tensorrt DEB package, the headers should in `/usr/include/x86_64-linux-gnu/`. If you install tensorrt TAR or ZIP file, the `include_directories` and `link_directories` of tensorrt should be added in `CMakeLists.txt`.

`dpkg -L` can print out the contents of a DEB package.

```
$ dpkg -L libnvinfer-dev 
/.
/usr
/usr/lib
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libnvinfer_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_compiler_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_executor_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_pattern_library_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_pattern_runtime_static.a
/usr/include
/usr/include/x86_64-linux-gnu
/usr/include/x86_64-linux-gnu/NvInfer.h
/usr/include/x86_64-linux-gnu/NvInferRuntime.h
/usr/include/x86_64-linux-gnu/NvInferRuntimeCommon.h
/usr/include/x86_64-linux-gnu/NvInferVersion.h
/usr/include/x86_64-linux-gnu/NvUtils.h
/usr/share
/usr/share/doc
/usr/share/doc/libnvinfer-dev
/usr/share/doc/libnvinfer-dev/copyright
/usr/share/doc/libnvinfer-dev/changelog.Debian
/usr/lib/x86_64-linux-gnu/libmyelin.so
/usr/lib/x86_64-linux-gnu/libnvinfer.so
```

## 2. fatal error: cuda_runtime_api.h: No such file or directory

`cuda_runtime_api.h` is from cuda-cudart. If you met this error, you need find where it is and adapt the `include_directories` and `link_directories` of cuda in `CMakeLists.txt`.

```
$ dpkg -L cuda-cudart-dev-10-0 
/.
/usr
/usr/local
/usr/local/cuda-10.0
/usr/local/cuda-10.0/targets
/usr/local/cuda-10.0/targets/x86_64-linux
/usr/local/cuda-10.0/targets/x86_64-linux/lib
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudadevrt.a
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libOpenCL.so.1.1
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libculibos.a
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart_static.a
/usr/local/cuda-10.0/targets/x86_64-linux/include
/usr/local/cuda-10.0/targets/x86_64-linux/include/cuda_runtime_api.h
/usr/local/cuda-10.0/targets/x86_64-linux/include/cudart_platform.h
/usr/local/cuda-10.0/targets/x86_64-linux/include/cuda_device_runtime_api.h
/usr/local/cuda-10.0/targets/x86_64-linux/include/cuda_runtime.h
/usr/lib
/usr/lib/pkgconfig
/usr/lib/pkgconfig/cudart-10.0.pc
/usr/share
/usr/share/doc
/usr/share/doc/cuda-cudart-dev-10-0
/usr/share/doc/cuda-cudart-dev-10-0/changelog.Debian.gz
/usr/share/doc/cuda-cudart-dev-10-0/copyright
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libOpenCL.so
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libOpenCL.so.1
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so
```

## 3. .wts not prepared or not in the right directory

If .wts file not in the right directory. The loadWeights() function will report error. Error logs like following.

By default, the .wts file usually should be put in the same dir as `build`. For example, `tensorrtx/yolov5/yolov5s.wts`. And the .wts path defined in `yolov5.cpp`.

```
std::map<std::__cxx11::basic_string, nvinfer1::Weights> loadWeights(std::__cxx11::string): Assertion `input.is_open() && "Unable to load weight file."' failed.
Aborted (core dumped)
```

## 4. yolo -s failed, class_num not adapted

If you train your own yolo model, you need set the `CLASS_NUM` in `yololayer.h`. Which is `80` by default. Otherwise, you will get errors like following.

```
[Convolution]: kernel weights has count xxx but xxx was expected
void APIToModel(unsigned int, nvinfer1::IHostMemory**): Assertion `engine != nullptr' failed.
Aborted (core dumped)
```

