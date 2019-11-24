# vgg

VGG 11-layer model (configuration "A") from
    "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>

For the Pytorch implementation, you can refer to [pytorchx/vgg](https://github.com/wang-xinyu/pytorchx/tree/master/vgg)

VGG's architecture is simple, just some conv, relu, maxpool, and fc layers.

```
// 1. generate vgg.wts from [pytorchx/vgg](https://github.com/wang-xinyu/pytorchx/tree/master/vgg)

// 2. put vgg.wts into tensorrtx/vgg

// 3. build and run

cd tensorrtx/vgg

mkdir build

cd build

cmake ..

make

sudo ./vgg -s   // serialize model to plan file i.e. 'vgg.engine'
sudo ./vgg -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/vgg
```


