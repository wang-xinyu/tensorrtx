# UNet

Pytorch model from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

## Contributors

<a href="https://github.com/YuzhouPeng"><img src="https://avatars.githubusercontent.com/u/13601004?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/East-Face"><img src="https://avatars.githubusercontent.com/u/35283869?v=4s=48" width="40px;" alt=""/></a>
<a href="https://github.com/irvingzhang0512"><img src="https://avatars.githubusercontent.com/u/22089207?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/wang-xinyu"><img src="https://avatars.githubusercontent.com/u/15235574?s=48&v=4" width="40px;" alt=""/></a>
<a href="https://github.com/nengwp"><img src="https://avatars.githubusercontent.com/u/44516353?s=96&v=4" width="40px;" alt=""/></a>


## Requirements

Now TensorRT 8.x is supported and you can use it.
The key cause of the previous bug is the pooling layer Stride setting problem.

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

4. Check result.jpg

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/207358769-dacf908e-f65d-4b2e-bc53-4fa2a9114c2a.jpg" height="360px;">
</p>

# Benchmark

Pytorch | TensorRT FP32 | TensorRT FP16
---- | ----- | ------ 
816x672  | 816x672 | 816x672
58ms  | 43ms (batchsize 8) | 14ms (batchsize 8)

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

