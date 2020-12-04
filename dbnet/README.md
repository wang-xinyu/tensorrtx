# DBNet

The Pytorch implementation is [DBNet](https://github.com/BaofengZan/DBNet.pytorch).

<p align="center">
<img src="https://user-images.githubusercontent.com/20653176/100968101-b044be00-356b-11eb-808c-9597cbe1f8de.jpg">
</p>


## How to Run

* 1. generate .wts

  Download code and model from [DBNet](https://github.com/BaofengZan/DBNet.pytorch) and config your environments.

  In tools/predict.py, set `save_wts` as `True`, and run, the .wts will be generated.

  onnx can also be exported, just need to set `onnx` as `True`.

* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  sudo ./dbnet -s             // serialize model to plan file i.e. 'DBNet.engine'
  sudo ./dbnet -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
  ```


## For windows

https://github.com/BaofengZan/DBNet-TensorRT

## Todo

* 1. ~~In common.hpp, the following two functions can be merged.~~

```c++
ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, bool bias = true) 
```

```c++
ILayer* convBnLeaky2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, bool bias = true)
```

* 2. The postprocess method here should be optimized, which is a little different from pytorch side.

* 3. ~~The input image here is resized to 640x640 directly, while the pytorch side is using `letterbox` method.~~

