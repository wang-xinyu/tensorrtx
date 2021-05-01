# HRNet-Semantic-Segmentation

This repo implemtents [HRNet-Semantic-Segmentation-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1) and [HRNet-Semantic-Segmentation-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).


## How to Run
### For HRNet-Semantic-Segmentation-v1.1
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
  ./hrnet -s [.wts] [.engine] [small or 18 or 32 or 48] # small for W18-Small-v2, 18 for W18, etc.
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
### For HRNet-Semantic-Segmentation-OCR

1. generate .wts, use config `experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml` and pretrained weight `hrnet_ocr_cs_8162_torch11.pth` as example. change `PRETRAINED` in `experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml` to `""`.
```
cp gen_wts.py $HRNET-OCR-TRAIN-PROJECT-ROOT/tools
cd $HRNET-OCR-PROJECT-ROOT
python tools/gen_wts.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml --ckpt_path hrnet_ocr_cs_8162_torch11.pth --save_path hrnet_ocr_w48.wts
cp hrnet_ocr_w48.wts $HRNET-OCR-TENSORRT-ROOT
cd $HRNET-OCR-TENSORRT-ROOT
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
  ./hrnet_ocr -s [.wts] [.engine] [18 or 32 or 48]
  ```
  such as
  ```
  ./hrnet_ocr -s ../hrnet_ocr_w48.wts ./hrnet_ocr_w48.engine 48
  ```
  then deserialize plan file and run inference
  ```
  ./hrnet_ocr -d  [.engine] [image dir]
  ```
  such as 
  ```
  ./hrnet_ocr -d  ./hrnet_ocr_w48.engine ../samples
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

