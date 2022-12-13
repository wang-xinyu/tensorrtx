# UNet

Pytorch model from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

## Requirements

Please use TensorRT 7.x.

There is a bug with TensorRT 8.x, we are working on it.

## Build and Run

1. Generate .wts
```
cp {path-of-tensorrtx}/unet/gen_wts.py Pytorch-UNet/
cd Pytorch-UNet/
wget https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth
python gen_wts.py unet_carvana_scale0.5_epoch2.pth
```

2. Generate TensorRT engine
```
cd tensorrtx/unet/
mkdir build
cd build
cmake ..
make
cp {path-of-Pytorch-UNet}/unet.wts .
./unet -s
```

3. Run inference
```
wget https://raw.githubusercontent.com/wang-xinyu/tensorrtx/f60dcc7bec28846cd973fc95ac829c4e57a11395/unet/samples/0cdf5b5d0ce1_01.jpg
./unet -d 0cdf5b5d0ce1_01.jpg
```

# Benchmark

Pytorch | TensorRT FP32 | TensorRT FP16
---- | ----- | ------ 
816x672  | 816x672 | 816x672
58ms  | 43ms (batchsize 8) | 14ms (batchsize 8)

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

