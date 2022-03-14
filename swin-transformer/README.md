# swin_transform

The Pytorch implementation is [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer.git).


## How to Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
git clone https://github.com/microsoft/Swin-Transformer.git
git clone https://github.com/wang-xinyu/tensorrtx.git

python gen_wts.py best.pt
// a file 'Swin-Transform.wts' will be generated.
```

2. build tensorrtx/Swin-Transform and run

```
cd {tensorrtx}/Swin-Transform/
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/Swin-Transform/Swin-Transform.wts {tensorrtx}/yolov5/build
cmake ..
make
sudo ./swintransformer -s [.wts] [.engine]   // serialize model to plan file
sudo ./swintransformer -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.

```


# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov5/build

3. set the macro `USE_INT8` in yolov5.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

