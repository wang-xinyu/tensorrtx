# HRNet

The Pytorch implementation is [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification).  The implemented model is **HRNet-W18-C-Small-v2** 


## How to Run

* 1. generate .wts

  Download code and model from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification) and config your environments.

  Put `demo.py`  in the `YOUR_ROOT_DIR\HRNet-Image-Classification\tools `  folder, set `savewts in  main()` as `True`, and run, the .wts will be generated.

* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  sudo ./hrnet -s             // serialize model to plan file i.e. 'hrnet.engine'
  sudo ./hrnet -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
  ```

## Result

The test img:

![](https://user-images.githubusercontent.com/20653176/93732833-ac103200-fc05-11ea-88ff-6f59f316a377.JPEG)

Pytorch Result:

![image-20200921115119593](https://user-images.githubusercontent.com/20653176/93731787-225e6580-fc01-11ea-9578-393079cd1873.png)

TRT Result:

![image-20200921114959069](https://user-images.githubusercontent.com/20653176/93731788-238f9280-fc01-11ea-954f-2debc20e102a.png)
