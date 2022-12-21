# YOLOv5

TensorRTx inference code base for [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## Contributors

<a href="https://github.com/wang-xinyu"><img src="https://avatars.githubusercontent.com/u/15235574?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/BaofengZan"><img src="https://avatars.githubusercontent.com/u/20653176?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/upczww"><img src="https://avatars.githubusercontent.com/u/16224249?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/cesarandreslopez"><img src="https://avatars.githubusercontent.com/u/14029177?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/makaveli10"><img src="https://avatars.githubusercontent.com/u/39617050?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/priteshgohil"><img src="https://avatars.githubusercontent.com/u/43172056?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/rymzt"><img src="https://avatars.githubusercontent.com/u/3270954?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/AsakusaRinne"><img src="https://avatars.githubusercontent.com/u/47343601?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/freedenS"><img src="https://avatars.githubusercontent.com/u/26213470?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/smarttowel"><img src="https://avatars.githubusercontent.com/u/1128528?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/wwqgtxx"><img src="https://avatars.githubusercontent.com/u/582584?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/adujardin"><img src="https://avatars.githubusercontent.com/u/12609780?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/jow905"><img src="https://avatars.githubusercontent.com/u/19189198?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/CristiFati"><img src="https://avatars.githubusercontent.com/u/29705787?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/HaiyangPeng"><img src="https://avatars.githubusercontent.com/u/46739135?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/Armassarion"><img src="https://avatars.githubusercontent.com/u/33727511?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/xupengao"><img src="https://avatars.githubusercontent.com/u/51817015?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/liuqi123123"><img src="https://avatars.githubusercontent.com/u/46275888?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/ASONG0506"><img src="https://avatars.githubusercontent.com/u/26050577?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/bobo0810"><img src="https://avatars.githubusercontent.com/u/26057879?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/Silmeria112"><img src="https://avatars.githubusercontent.com/u/16464837?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/LW-SCU"><img src="https://avatars.githubusercontent.com/u/28128257?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/AdanWang"><img src="https://avatars.githubusercontent.com/u/32757980?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/triple-Mu"><img src="https://avatars.githubusercontent.com/u/92794867?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/xiang-wuu"><img src="https://avatars.githubusercontent.com/u/107029401?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/uyolo1314"><img src="https://avatars.githubusercontent.com/u/101853326?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/Rex-LK"><img src="https://avatars.githubusercontent.com/u/74702576?s=96&v=4" width="40px;" alt=""/></a>

## Different versions of yolov5

Currently, we support yolov5 v1.0, v2.0, v3.0, v3.1, v4.0, v5.0, v6.0, v6.2, v7.0

- For yolov5 v7.0, download .pt from [yolov5 release v7.0](https://github.com/ultralytics/yolov5/releases/tag/v7.0), `git clone -b v7.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v7.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v7.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v7.0/yolov5)
- For yolov5 v6.2, download .pt from [yolov5 release v6.2](https://github.com/ultralytics/yolov5/releases/tag/v6.2), `git clone -b v6.2 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v6.2 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v6.2](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v6.2/yolov5)
- For yolov5 v6.0, download .pt from [yolov5 release v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0), `git clone -b v6.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v6.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v6.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v6.0/yolov5).
- For yolov5 v5.0, download .pt from [yolov5 release v5.0](https://github.com/ultralytics/yolov5/releases/tag/v5.0), `git clone -b v5.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v5.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v5.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v5.0/yolov5).
- For yolov5 v4.0, download .pt from [yolov5 release v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0), `git clone -b v4.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v4.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v4.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v4.0/yolov5).
- For yolov5 v3.1, download .pt from [yolov5 release v3.1](https://github.com/ultralytics/yolov5/releases/tag/v3.1), `git clone -b v3.1 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v3.1 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v3.1](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v3.1/yolov5).
- For yolov5 v3.0, download .pt from [yolov5 release v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0), `git clone -b v3.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v3.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v3.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v3.0/yolov5).
- For yolov5 v2.0, download .pt from [yolov5 release v2.0](https://github.com/ultralytics/yolov5/releases/tag/v2.0), `git clone -b v2.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v2.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v2.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v2.0/yolov5).
- For yolov5 v1.0, download .pt from [yolov5 release v1.0](https://github.com/ultralytics/yolov5/releases/tag/v1.0), `git clone -b v1.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v1.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v1.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v1.0/yolov5).

## Config

- Choose the model n/s/m/l/x/n6/s6/m6/l6/x6 from command line arguments.
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in yolov5.cpp, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp
- Batch size in yolov5.cpp

## Build and Run

### Detection

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
// clone code according to above #Different versions of yolov5
// download https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
cp {tensorrtx}/yolov5/gen_wts.py {ultralytics}/yolov5
cd {ultralytics}/yolov5
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
// a file 'yolov5s.wts' will be generated.
```

2. build tensorrtx/yolov5 and run

```
cd {tensorrtx}/yolov5/
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
cmake ..
make
sudo ./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
sudo ./yolov5_det -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov5s
sudo ./yolov5_det -s yolov5s.wts yolov5s.engine s
sudo ./yolov5_det -d yolov5s.engine ../samples
// For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
sudo ./yolov5_det -s yolov5_custom.wts yolov5.engine c 0.17 0.25
sudo ./yolov5_det -d yolov5.engine ../samples
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov5s.engine and libmyplugins.so have been built
python yolov5_det_trt.py

// Another version of python script, which is using CUDA Python instead of pycuda.
python yolov5_det_trt_cuda_python.py
```

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg" height="360px;">
</p>

### Classification

```
# Download ImageNet labels
wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt

# Build and serialize TensorRT engine
./yolov5_cls -s yolov5s-cls.wts yolov5s-cls.engine s

# Run inference
./yolov5_cls -d yolov5s-cls.engine ../samples
```

### Instance Segmentation

```
# Build and serialize TensorRT engine
./yolov5_seg -s yolov5s-seg.wts yolov5s-seg.engine s

# Run inference
./yolov5_seg -d yolov5s-seg.engine ../samples
```

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/208305921-0a2ee358-6550-4d36-bb86-867685bfe069.jpg" height="360px;">
</p>

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov5/build

3. set the macro `USE_INT8` in yolov5.cpp and make

4. serialize the model and test


## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

