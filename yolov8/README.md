# YOLOv8

The Pytorch implementation is [ultralytics/yolov8](https://github.com/ultralytics/ultralytics/tree/main/ultralytics).

The tensorrt code is derived from [xiaocao-tian/yolov8_tensorrt](https://github.com/xiaocao-tian/yolov8_tensorrt)

## Contributors

<a href="https://github.com/xiaocao-tian"><img src="https://avatars.githubusercontent.com/u/65889782?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/lindsayshuo"><img src="https://avatars.githubusercontent.com/u/45239466?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/xinsuinizhuan"><img src="https://avatars.githubusercontent.com/u/40679769?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/Rex-LK"><img src="https://avatars.githubusercontent.com/u/74702576?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/emptysoal"><img src="https://avatars.githubusercontent.com/u/57931586?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/ChangjunDAI"><img src="https://avatars.githubusercontent.com/u/65420228?s=48&v=4" width="40px;" alt=""/></a>

## Requirements

- TensorRT 8.0+
- OpenCV 3.4.0+

## Different versions of yolov8

Currently, we support yolov8

- For yolov8 , download .pt from [https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases), then follow how-to-run in current page.
## Config

- Choose the model n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6 from command line arguments.
- Check more configs in [include/config.h](./include/config.h)

## How to Run, yolov8n as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
// download https://github.com/ultralytics/assets/releases/yolov8n.pt
// download https://github.com/lindsayshuo/yolov8-p2/releases/download/VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.pt (only for 10 cls p2 model)
cp {tensorrtx}/yolov8/gen_wts.py {ultralytics}/ultralytics
cd {ultralytics}/ultralytics
python gen_wts.py -w yolov8n.pt -o yolov8n.wts -t detect
// a file 'yolov8n.wts' will be generated.


// For p2 model
// download https://github.com/lindsayshuo/yolov8_p2_tensorrtx/releases/download/VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last/VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.pt (only for 10 cls p2 model)
python gen_wts.py -w VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.pt -o VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.wts -t detect (only for  10 cls p2 model)
// a file 'VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.wts' will be generated.
```

2. build tensorrtx/yolov8 and run

### Detection
```
cd {tensorrtx}/yolov8/
// update kNumClass in config.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/ultralytics/yolov8.wts {tensorrtx}/yolov8/build
cmake ..
make
sudo ./yolov8_det -s [.wts] [.engine] [n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6]  // serialize model to plan file
sudo ./yolov8_det -d [.engine] [image folder]  [c/g] // deserialize and run inference, the images in [image folder] will be processed.

// For example yolov8n
sudo ./yolov8_det -s yolov8n.wts yolov8.engine n
sudo ./yolov8_det -d yolov8n.engine ../images c //cpu postprocess
sudo ./yolov8_det -d yolov8n.engine ../images g //gpu postprocess


// For p2 model:
// change the  "const static int kNumClass" in config.h to 10;
sudo ./yolov8_det -s VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.wts VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.engine x2
wget https://github.com/lindsayshuo/yolov8-p2/releases/download/VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last/0000008_01999_d_0000040.jpg
cp -r 0000008_01999_d_0000040.jpg ../images
sudo ./yolov8_det -d VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.engine ../images c //cpu postprocess
sudo ./yolov8_det -d VisDrone_train_yolov8x_p2_bs1_epochs_100_imgsz_1280_last.engine ../images g //gpu postprocess
```

### Instance Segmentation
```
# Build and serialize TensorRT engine
./yolov8_seg -s yolov8s-seg.wts yolov8s-seg.engine s

# Download the labels file
wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt

# Run inference with labels file
./yolov8_seg -d yolov8s-seg.engine ../images c coco.txt
```

### Classification
```
cd {tensorrtx}/yolov8/
// Download inference images
wget  https://github.com/lindsayshuo/infer_pic/blob/main/1709970363.6990473rescls.jpg
mkdir samples
cp -r  1709970363.6990473rescls.jpg samples
// Download ImageNet labels
wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt

// update kClsNumClass in config.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/ultralytics/yolov8n-cls.wts {tensorrtx}/yolov8/build
cmake ..
make
sudo ./yolov8_cls -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file
sudo ./yolov8_cls -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.

// For example yolov8n
sudo ./yolov8_cls -s yolov8n-cls.wts yolov8-cls.engine n
sudo ./yolov8_cls -d yolov8n-cls.engine ../samples
```


### Pose Estimation
```
cd {tensorrtx}/yolov8/
// update "kNumClass = 1" in config.h
mkdir build
cd build
cp {ultralytics}/ultralytics/yolov8-pose.wts {tensorrtx}/yolov8/build
cmake ..
make
sudo ./yolov8_pose -s [.wts] [.engine] [n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6]  // serialize model to plan file
sudo ./yolov8_pose -d [.engine] [image folder]  [c/g] // deserialize and run inference, the images in [image folder] will be processed.

// For example yolov8-pose
sudo ./yolov8_pose -s yolov8n-pose.wts yolov8n-pose.engine n
sudo ./yolov8_pose -d yolov8n-pose.engine ../images c //cpu postprocess
sudo ./yolov8_pose -d yolov8n-pose.engine ../images g //gpu postprocess
```


4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov8n.engine and libmyplugins.so have been built
python yolov8_det_trt.py  # Detection
python yolov8_seg_trt.py  # Segmentation
python yolov8_cls_trt.py  # Classification
python yolov8_pose_trt.py  # Pose Estimation
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov8/build

3. set the macro `USE_INT8` in config.h, change `kInputQuantizationFolder` into your image folder path and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg" height="360px;">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
