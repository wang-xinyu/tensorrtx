# How to Compile and Run on Windows

This tutorial can be applied to any models in this repo. Only need to adapt couple of lines.

## Environments

* vs (only vs2015, vs2017 tested)
* cuda
* TensorRT
* Cmake
* opencv
* dirent.h for windows, put into tensorrtx/include, download from https://github.com/tronkko/dirent

  ![image-20200828131208257](https://user-images.githubusercontent.com/20653176/91524367-99217f00-e931-11ea-9a13-fb420403b73b.png)

## Compile and Run

### 1. Modify CmakeLists.txt

```cmake
cmake_minimum_required(VERSION 2.6)

project(yolov5) # 1
set(OpenCV_DIR "D:\\opencv\\opencv346\\build")  #2
set(TRT_DIR "D:\\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6\\TensorRT-7.0.0.11")  #3

add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}\\include)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
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
link_directories(${TRT_DIR}\\lib)

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yololayer.h)   #4

target_link_libraries(yolov5 "nvinfer" "nvinfer_plugin")   #5
target_link_libraries(yolov5 ${OpenCV_LIBS})          #6
target_link_libraries(yolov5 ${CUDA_LIBRARIES})   #7
target_link_libraries(yolov5 Threads::Threads)       #8
```

Notice: 8 lines to adapt in CMakeLists.txt, marked with #1-#8

- #1 project name, set according to your project name
- #2 your opencv path
- #3 your tensorrt path
- #4 source file needed, including .cpp .cu .h
- #5-#8 libs needed

### 2. run cmake-gui to config the project

#### 2.1 open cmake-gui and set the path

![image-20200828124434245](https://user-images.githubusercontent.com/20653176/91524158-1dbfcd80-e931-11ea-8a82-518eaf391d5a.png)

#### 2.2 click **Configure** and set the envs

![image-20200828124902923](https://user-images.githubusercontent.com/20653176/91524303-75f6cf80-e931-11ea-8591-64a8a1a9292b.png)

#### 2.3 click **Finish**, and wait for the `Configuring done`

![image-20200828124951872](https://user-images.githubusercontent.com/20653176/91524340-8b6bf980-e931-11ea-9ea4-141f5b94aa0a.png)

#### 2.4 click **Generate**

![image-20200828125046738](https://user-images.githubusercontent.com/20653176/91524350-8eff8080-e931-11ea-9ed1-82c5af2f558f.png)

#### 2.5 click **Open Project**

![image-20200828125215067](https://user-images.githubusercontent.com/20653176/91524352-9030ad80-e931-11ea-877e-dc08bfaef731.png)

#### 2.6 Click **Generate -> Generate solution**

![image-20200828125402056](https://user-images.githubusercontent.com/20653176/91524356-9161da80-e931-11ea-84ba-177e12200e04.png)

### 3. run in command line

cd to the path of exe (e.g. E:\LearningCodes\GithubRepo\tensorrtx\yolov5\build\Debug)

```
yolov5.exe -s             // serialize model to plan file i.e. 'yolov5s.engine'
yolov5.exe -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
```

**Notice**: while serializing the model, the .wts should put in the parent dir of xxx.vcxproj, or just modify the .wts path in yolov5.cpp

![image-20200828125938472](https://user-images.githubusercontent.com/20653176/91524358-93c43480-e931-11ea-81b6-ae01b92e1146.png)

### 4. run in vs

In vs, firstly `Set As Startup Project`, and then setup `Project ==> Properties ==> Configuration Properties ==> Debugging ==> Command Arguments` as `-s` or `-d ../yolov3-spp/samples`. Then can run or debug.

![image-20200828130117902](https://user-images.githubusercontent.com/20653176/91524360-94f56180-e931-11ea-9873-39bed7ee19f1.png)

![image-20200828130415658](https://user-images.githubusercontent.com/20653176/91524362-96bf2500-e931-11ea-8c79-8db3a25fc135.png)

![image-20200828131516231](https://user-images.githubusercontent.com/20653176/91524370-9a52ac00-e931-11ea-8c1a-acf828fe81b4.png)

**Notice**: The .dll of tensorrt and opencv should be put in the same directory with exe file. Or set environment variables in windows.(Not recommended)
