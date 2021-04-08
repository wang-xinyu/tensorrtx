# DBNet

The Pytorch implementation is [DBNet](https://github.com/BaofengZan/DBNet.pytorch).

<p align="center">
<img src="https://user-images.githubusercontent.com/25873202/113959270-1eb8c600-9855-11eb-9c4d-1e6dc8e38a17.jpg">
</p>



## How to Run

* 1. generate `.wts`

  Download code and model from [DBNet](https://github.com/BaofengZan/DBNet.pytorch) and config your environments.

  Go to file`tools/predict.py`, set `--save_wts` as `True`, then run, the `DBNet.wts` will be generated.

  Onnx can also be exported, just need to set `--onnx` as `True`.

* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  cp /your_wts_path/DBNet.wts .
  sudo ./dbnet -s             // serialize model to plan file i.e. 'DBNet.engine'
  sudo ./dbnet -d  ./test_imgs // deserialize plan file and run inference, all images in test_imgs folder will be processed.
  ```



## For windows

https://github.com/BaofengZan/DBNet-TensorRT



## Todo

- [x] 1. In `common.hpp`, the following two functions can be merged.

     ```c++
     ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, bool bias = true) 
     ```

     ```c++
     ILayer* convBnLeaky2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, bool bias = true)
     ```

- [x] 2. The postprocess method here should be optimized, which is a little different from pytorch side.

- [x] 3. The input image here is resized to `640 x 640` directly, while the pytorch side is using `letterbox` method.

