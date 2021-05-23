# Inception v3

Inception v3 model architecture from "Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>.

For the details, you can refer to [pytorchx/inception](https://github.com/wang-xinyu/pytorchx/tree/master/inception)

Following tricks are used in this inception:

- For pooling layer with padding, we need pay attention to see if padding is included or excluded while calculating average number. Pytorch includes padding while doing avgPool by default, but Tensorrt doesn't. So for pooling layer with padding, we need `setAverageCountExcludesPadding(false)` in tensorrt.
- Batchnorm layer, implemented by scale layer.

```
// 1. generate inception.wts from [pytorchx/inception](https://github.com/wang-xinyu/pytorchx/tree/master/inception)

// 2. put inception.wts into tensorrtx/inception

// 3. build and run

cd tensorrtx/inception

mkdir build

cd build

cmake ..

make

sudo ./inception -s   // serialize model to plan file i.e. 'inception.engine'

sudo ./inception -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/inception
```


