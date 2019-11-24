# googlenet

GoogLeNet (Inception v1) model architecture from "Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

For the details, you can refer to [pytorchx/googlenet](https://github.com/wang-xinyu/pytorchx/tree/master/googlenet)

Following tricks used in this googlenet:

- MaxPool2d(ceil_mode=True), ceilmode=True, which is not supported in Tensorrt4, we use a padding layer before maxpool to solve this problem.
- Batchnorm layer, implemented by scale layer.

```
// 1. generate googlenet.wts from [pytorchx/googlenet](https://github.com/wang-xinyu/pytorchx/tree/master/googlenet)

// 2. put googlenet.wts into tensorrtx/googlenet

// 3. build and run

cd tensorrtx/googlenet

mkdir build

cd build

cmake ..

make

sudo ./googlenet -s   // serialize model to plan file i.e. 'googlenet.engine'

sudo ./googlenet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/googlenet
```


