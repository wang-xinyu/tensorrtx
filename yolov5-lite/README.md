# YOLOv5-Lite TensorRT Deployment



Detection training code [link](https://github.com/ppogg/YOLOv5-Lite.git)

## Environment
TensorRT: 8.6.1.6
CUDA: 12.6
CUDNN: 8.9.0
OpenCV:4.10.0



## Configuration parameters

Before starting, you need to modify parameters in `include/yololayer.h` to match your training configuration (example at `include/yololayer.h`):

```cpp
static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
static constexpr int CLASS_NUM = 80;  // number of classes
static constexpr int INPUT_H = 640;   // input height for yolov5-lite (must be divisible by 32)
static constexpr int INPUT_W = 640;   // input width for yolov5-lite (must be divisible by 32)
static constexpr int DEVICE = 0;
static constexpr float NMS_THRESH = 0.4;
static constexpr float CONF_THRESH = 0.45;
static constexpr int BATCH_SIZE = 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
```

## 1. Generate .wts from .pt

This step must be performed inside the `yolov5-lite` folder:

```bash
cd yolov5-lite
git clone https://gitcode.com/open-source-toolkit/ac70a.git
unzip your zip file 

python gen_wts.py -w v5lite-s.pt -o v5lite-s.wts
python gen_wts.py -w v5lite-e.pt -o v5lite-e.wts
python gen_wts.py -w v5lite-g.pt -o v5lite-g.wts
```

## 2. Build the engine and run inference

### Build steps

a. First, set `CLASS_NUM` in `include/yololayer.h` to match your dataset class count — this is important, otherwise you will get errors.

b. Run the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

### Generate engine files

```bash
./v5lite -s ../v5lite-s.wts v5lite-s.engine s
./v5lite -s ../v5lite-g.wts v5lite-g.engine g
./v5lite -s ../v5lite-e.wts v5lite-e.engine e
./v5lite -s ../v5lite-c.wts v5lite-c.engine c
```

### Using the engine for inference

(`samples` is the folder containing your images):

```bash
./v5lite -d v5lite-s.engine ../samples
```

You can also use `yolov5-lite-trt.py` (in the repository root) for inference.

## 3. INT8 Quantization

### Preparation

1. Collect calibration images (recommended ~1000 images)
2. Put the images in a calibration folder (for example: `tensorrtx-int8calib-data/coco_calib`)
3. Modify the macro in [v5lite.cpp](yolov5-lite/v5lite.cpp):

   Change:
   ```cpp
   // #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
   // #define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32
   ```

   To:
   ```cpp
   // #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
   #define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32
   ```

4. Update the data path in the code to point to your calibration images

5. Rebuild and generate the engine, then run inference (repeat step 2)

## Notes

- In practice, calling the engine from Python may produce better inference behavior in some cases.
