# Inception v4

Inception v4 model architecture from "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" <https://arxiv.org/abs/1602.07261v2>.

For the details, you can refer to [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v4.py)

Following tricks are used in this inception:

- For pooling layer with padding, we need pay attention to see if padding is included or excluded while calculating average number. Pytorch includes padding while doing avgPool by default, but Tensorrt doesn't. So for pooling layer with padding, we need `setAverageCountExcludesPadding(false)` in tensorrt.
- Batchnorm layer, implemented by scale layer.

```
// 1. generate inception.wts from [BlueMirrors/torchtrtz](https://github.com/BlueMirrors/torchtrtz/blob/main/generate_weights.py)

// 2. put inception.wts into tensorrtx/inceptionV4

// 3. build and run

cd tensorrtx/inception/inceptionV4

mkdir build

cd build

cmake ..

make

sudo ./inceptionV4 -s   // serialize model to plan file i.e. 'inceptionV4.engine'

sudo ./inceptionV4 -d   // deserialize plan file and run inference

// 4. see if the output is same as rwightman/pytorch-image-models/inceptionv4
```


