# TensorRT implementation for HRNet+OCR

The Pytorch implementation is [HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).  The implemented modeld include **HRNetV2-W18 + OCR, HRNetV2-W32 + OCR, HRNetV2-W48 + OCR** or any width.


## How to Run

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

Result:



## Note

* Image preprocessing operation and postprocessing operation are put into Trt Engine.

* Zero-copy technology (CPU/GPU memory copy) is used.

