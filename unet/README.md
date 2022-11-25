# UNet
This is a TensorRT version UNet, inspired by [tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [pytorch-unet](https://github.com/milesial/Pytorch-UNet).<br>
You can generate TensorRT engine file using this script and customize some params and network structure based on network you trained (FP32/16 precision, input size, different conv, activation function...)<br>

# Requirements

TensorRT 7.x or 8.x (you need to install tensorrt first)<br>
Python<br>
opencv<br>
cmake<br>

# Train .pth file and convert .wts

## Create env

```
pip install -r requirements.txt
```

## Train .pth file

Train your dataset by following [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) and generate .pth file.<br>

Please set bilinear=False, i.e. `UNet(n_channels=3, n_classes=1, bilinear=False)`, because TensorRT doesn't support Upsample layer.

## Convert .pth to .wts

```
cp tensorrtx/unet/gen_wts.py Pytorch-UNet/
cd Pytorch-UNet/
python gen_wts.py
```

# Generate engine file and infer

Build:
```
cd tensorrtx/unet/
mkdir build
cd build
cmake ..
make
```

Generate TensorRT engine file:
```
unet -s
```
Inference on images in a folder:
```
unet -d ../samples
```

# Benchmark
the speed of tensorRT engine is much faster

 pytorch | TensorRT FP32 | TensorRT FP16
 ---- | ----- | ------  
 816x672  | 816x672 | 816x672 
 58ms  | 43ms (batchsize 8) | 14ms (batchsize 8) 


# Further development

1. add INT8 calibrator<br>
2. add custom plugin<br>
