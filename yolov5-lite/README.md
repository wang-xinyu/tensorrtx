# YOLOv5-Lite TensorRT Deployment

## Environment
TensorRT: 8.6.1.6
CUDA: 12.6
CUDNN: 8.9.0


## 配置参数修改

在开始之前，需要修改 `include/yololayer.h` 文件中的相关参数，确保它们与您的训练过程配置一致（示例位于 `include/yololayer.h`）：

```cpp
static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000; 
static constexpr int CLASS_NUM = 80;  // 类别数
static constexpr int INPUT_H = 640;   // yolov5-lite的输入高度，必须能被32整除
static constexpr int INPUT_W = 640;   // yolov5-lite的输入宽度，必须能被32整除
static constexpr int DEVICE = 0;
static constexpr float NMS_THRESH = 0.4;
static constexpr float CONF_THRESH = 0.45;
static constexpr int BATCH_SIZE = 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
```

## 1. 由pt文件生成wts文件

此过程必须在 `yolov5-lite` 文件夹中完成：

```bash
python gen_wts.py -w v5lite-s.pt -o v5lite-s.wts
python gen_wts.py -w v5lite-e.pt -o v5lite-e.wts
python gen_wts.py -w v5lite-g.pt -o v5lite-g.wts
```

## 2. 编译engine文件和推理

### 编译步骤

a. 首先在 `include/yololayer.h` 文件中将 `CLASS_NUM` 改为您的数据集对应的类别数，这一步非常重要，否则会报错！

b. 在终端中执行以下命令：

```bash
mkdir build
cd build
cmake ..
make
```

### 生成engine文件

```bash
./v5lite -s ../v5lite-s.wts v5lite-s.engine s
./v5lite -s ../v5lite-g.wts v5lite-g.engine g
./v5lite -s ../v5lite-e.wts v5lite-e.engine e
./v5lite -s ../v5lite-c.wts v5lite-c.engine c
```

### 使用engine文件进行推理

(samples是您存放图片的文件夹)：

```bash
./v5lite -d v5lite-s.engine ../samples
```

您也可以使用 `yolov5-lite-trt.py`（位于仓库根目录）进行推理。

## 3. INT8量化

### 准备工作

1. 获取训练集的图片，数量建议1000张左右
2. 将图片放入指定的校准数据文件夹中（例如：`tensorrtx-int8calib-data/coco_calib`）
3. 修改[v5lite.cpp](yolov5-lite/v5lite.cpp)中的宏定义：
   
   将：
   ```cpp
   // #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
   // #define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32
   ```
   
   改为：
   ```cpp
   // #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
   #define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32
   ```

4. 更新代码中的数据路径，指向您的校准图像文件夹

5. 重新编译并生成engine进行推理(重复步骤2)

## 注意事项

- 实际测试中发现 Python 调用 engine 的推理效果可能更好

## 模型权重（.wts）管理建议

`.wts` 通常为较大的二进制文件。将这些大文件直接保存在源码仓库会导致仓库体积增大并影响 clone/checkout 速度。常用的替代方案：

- 把 `.wts` 上传到 GitHub Releases，并在 README 中提供下载链接（推荐）；
- 或者使用 Git LFS 管理大文件（需要在本地安装 Git LFS 并在 GitHub 启用）。示例：

```bash
# 在本地仓库中安装并初始化 git lfs（仅需一次）
git lfs install
# 在仓库中跟踪 .wts 文件（会在 .gitattributes 中添加规则）
git lfs track "yolov5-lite/*.wts"
# 提交 .gitattributes
git add .gitattributes
git commit -m "chore: track .wts with git lfs"
```

如果你想把已存在的 `.wts` 从 Git 的追踪中移除（保留本地文件并把权重放到 Releases 或迁移到 LFS），可以执行：

```bash
# 仅从 git 追踪中移除（保留本地副本）
git rm --cached yolov5-lite/v5lite-*.wts
git commit -m "chore: remove large .wts from git tracking; use Releases or Git LFS"
```

我将为你在仓库根目录添加一个 `.gitattributes`（包含 yolov5-lite 下 .wts 的 LFS 规则），并把当前 `yolov5-lite/v5lite-*.wts` 从 Git 追踪中移除（如果你确认执行）。