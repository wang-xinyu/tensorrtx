# windows编译运行手把手教程

## 说明

本教程可以用于https://github.com/wang-xinyu/tensorrtx该仓库中所有项目，只需要修改几处代码即可，修改之处后面会详细注明。

## 所需环境

* vs 版本可不限（仅测试了 vs2015 vs2017）

* cuda

* TensorRT需要和本机的cuda版本适配（需要7.0以上版本）

* Cmake 

* opencv （需要和vs版本适配）

* 增加windows下的文件夹处理头文件dirent.h，放在当前文件下include下。下载地址 https://github.com/tronkko/dirent

  ![image-20200828131208257](https://user-images.githubusercontent.com/20653176/91524367-99217f00-e931-11ea-9a13-fb420403b73b.png)

## 编译

### 修改CmakeLists.txt

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

#设置cuda信息
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)
####
enable_language(CUDA)  # 这一句添加后 ，就会在vs中不需要再手动设置cuda 
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}\\include)

#-D_MWAITXINTRIN_H_INCLUDED  解决error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

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
link_directories(${TRT_DIR}\\lib)

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yololayer.h 
                ${PROJECT_SOURCE_DIR}/hardswish.cu ${PROJECT_SOURCE_DIR}/hardswish.h)   #4

target_link_libraries(yolov5  "nvinfer" "nvinfer_plugin")   #5
target_link_libraries(yolov5 ${OpenCV_LIBS})          #6
target_link_libraries(yolov5 ${CUDA_LIBRARIES})   #7
target_link_libraries(yolov5 Threads::Threads)       #8
```

注意：在CamakeLists.txt中，有8处需要修改，在#1-#8标注的位置

1. #1 为当前编译的工程名，根据你所编译的工程名修改，也可以自定义
2. #2 为本机的opencv地址
3. #3 为本机TensorRT的安装位置
4. #4 为当前编译工程中所涉及的文件，包括.cpp .cu .h等文件，另外第一个参数要与#1设置的相同
5. #5 -#8 为链接lib，工程名需要和#1设置的一样。

### cmake-gui编译

#### 1 打开cmake-gui，并设置相应路径：如下图

![image-20200828124434245](https://user-images.githubusercontent.com/20653176/91524158-1dbfcd80-e931-11ea-8a82-518eaf391d5a.png)

#### 点击1处**Configure**，弹出窗口，并选择相应环境：

![image-20200828124902923](https://user-images.githubusercontent.com/20653176/91524303-75f6cf80-e931-11ea-8591-64a8a1a9292b.png)

点击Finish完成设置，等待生成完成：

![image-20200828124951872](https://user-images.githubusercontent.com/20653176/91524340-8b6bf980-e931-11ea-9ea4-141f5b94aa0a.png)

#### 点击2处Generate

![image-20200828125046738](https://user-images.githubusercontent.com/20653176/91524350-8eff8080-e931-11ea-9ed1-82c5af2f558f.png)

#### 点击3处Open Project，打开工程

![image-20200828125215067](https://user-images.githubusercontent.com/20653176/91524352-9030ad80-e931-11ea-877e-dc08bfaef731.png)

## 运行

点击“生成-生成解决方案”。等待编译完成。

![image-20200828125402056](https://user-images.githubusercontent.com/20653176/91524356-9161da80-e931-11ea-84ba-177e12200e04.png)

### 运行方法1：命令行

命令行cd到exe的路径（比如：E:\LearningCodes\GithubRepo\tensorrtx\yolov5\build\Debug）,

```
yolov5.exe -s             // serialize model to plan file i.e. 'yolov5s.engine'
yolov5.exe -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
```

**注意：在生成engine时，wts文件要放置在工程文件xxx.vcxproj所在文件的上一级目录**，或者直接修改源码，直接指向绝对地址：

![image-20200828125938472](https://user-images.githubusercontent.com/20653176/91524358-93c43480-e931-11ea-81b6-ae01b92e1146.png)

### 运行方法2：vs直接运行，可debug

在vs中，在工程上右键，先选择“设为启动项”，然后选择“属性”。在属性页设置参数。设置完成后点击运行即可，可以打断点debug。

![image-20200828130117902](https://user-images.githubusercontent.com/20653176/91524360-94f56180-e931-11ea-9873-39bed7ee19f1.png)

![image-20200828130415658](https://user-images.githubusercontent.com/20653176/91524362-96bf2500-e931-11ea-8c79-8db3a25fc135.png)

![image-20200828131516231](https://user-images.githubusercontent.com/20653176/91524370-9a52ac00-e931-11ea-8c1a-acf828fe81b4.png)

**注意：**

**运行时需要将tensorRt的dll和opencv的dll拷贝到exe所在路径。或者将tensorRt和opencv的bin文件夹路径添加到环境变量中（不建议）**