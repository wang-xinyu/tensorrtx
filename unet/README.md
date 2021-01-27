# tensorrt-unet
This is a TensorRT version Unet, inspired by [tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [pytorch-unet](https://github.com/milesial/Pytorch-UNet).<br>
You can generate TensorRT engine file using this script and customize some params and network structure based on network you trained (FP32/16 precision, input size, different conv, activation function...)<br>

# requirements

TensorRT 7.0 (you need to install tensorrt first)<br>
Cuda 10.2<br>
Python3.7<br>
opencv 4.4<br>
cmake 3.18<br>
# train .pth file and convert .wts

## create env

```
pip install -r requirements.txt
```

## train .pth file

train your dataset by following [pytorch-unet](https://github.com/milesial/Pytorch-UNet) and generate .pth file.<br>

## convert .wts

run gen_wts from utils folder, and move it to project folder<br>

# generate engine file and infer

## create build folder in project folder
```
mkdir build
```

## make file, generate exec file
```
cd build
cmake ..
make
```

## generate TensorRT engine file and infer image
```
unet -s
```
then a unet exec file will generated, you can use unet -d to infer files in a folder<br>
```
unet -d ../samples
```

# efficiency
the speed of tensorRT engine is much faster

 pytorch | TensorRT FP32 | TensorRT FP16
 ---- | ----- | ------  
 816x672  | 816x672 | 816x672 
 58ms  | 43ms (batchsize 8) | 14ms (batchsize 8) 


# Further development

1. add INT8 calibrator<br>
2. add custom plugin<br>
etc
