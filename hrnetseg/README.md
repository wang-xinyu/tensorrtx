# HRNet-Semantic-Segmentation

The Pytorch implementation is [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation).  The implemented model is **HRNetV2-W18-Small-v2**


## How to Run

* 1. generate .wts

  Download code and model from [ HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Image-Classification) and config your environments.

  Put `demo.py`  in the `YOUR_ROOT_DIR\HRNet-Semantic-Segmentation\tools `  folder, set `savewts in  main()` as `True`, and run, the .wts will be generated.

* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  sudo ./hrnetseg -s             // serialize model to plan file i.e. 'hrnetseg.engine'
  sudo ./hrnetseg -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
  ```

## Result

TRT Result:

![trtcity](https://user-images.githubusercontent.com/20653176/103136469-a68e2080-46fb-11eb-9f05-06bad81c74b9.png)

pytorch result:

![image-20201225171224159](https://user-images.githubusercontent.com/20653176/103131619-6cf9ed00-46dc-11eb-9369-4374abb65744.png)

## Note

* Some source codes are changed for simplicity.  But the original model can still be used.

  All "upsample" op  in source code are changed to `mode='bilinear', align_corners=True`

* Image preprocessing operation and postprocessing operation  are put into Trt Engine.

* Zero-copy technology (CPU/GPU memory copy) is used.

