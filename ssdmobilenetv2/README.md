# ssd mobilenet v2

MobileNetV2 architecture from
     "MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>.

For the Pytorch implementation, you can refer to [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)

```
// 1. generate mobilenet.wts from [pytorchx/mobilenet](https://github.com/wang-xinyu/pytorchx/tree/master/mobilenet) using `gen_wts.py` script

// 2. put ssdmobilenet.wts into tensorrtx/ssd-mobilenetv2

// 3. build and run

cd tensorrtx/ssd-mobilenetv2

mkdir build

cd build

cmake ..

make

sudo ./ssd-mobilenetv2 -s   // serialize model to plan file i.e. 'ssd_mobilenet_v2.engine'

sudo ./ssd-mobilenetv2 -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/mobilenet
```
