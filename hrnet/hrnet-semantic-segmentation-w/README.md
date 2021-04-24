# TensorRT implementation for HRNet-Semantic-Segmentation

The Pytorch implementation is [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).  The implemented modeld include **HRNetV2-W18, HRNetV2-W32, HRNetV2-W48** or any width.


## How to Run

1. generate .wts, use config `experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml` and pretrained weight `hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth` as example. change `PRETRAINED` in `experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml` to `""`.
```
cp gen_wts.py $HRNET--Semantic-Segmentation-PROJECT-ROOT/tools
cd $HRNET--Semantic-Segmentation-PROJECT-ROOT
python tools/gen_wts.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml --ckpt_path hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth --save_path hrnet_w48.wts
cp hrnet_w48.wts $HRNET-TENSORRT-ROOT
cd $HRNET-TENSORRT-ROOT
```
2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
  first serialize model to plan file
  ```
  ./hrnet -s [.wts] [.engine] [18 or 32 or 48]
  ```
  such as
  ```
  ./hrnet -s ../hrnet_w48.wts ./hrnet_w48.engine 48
  ```
  then deserialize plan file and run inference
  ```
  ./hrnet -d  [.engine] [image dir]
  ```
  such as 
  ```
  ./hrnet -d  ./hrnet_w48.engine ../samples
  ```

## Result

Result:



## Note

* Image preprocessing operation and postprocessing operation are put into Trt Engine.

* Zero-copy technology (CPU/GPU memory copy) is used.

