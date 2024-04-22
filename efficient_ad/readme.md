# EfficientAd

EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

The Pytorch implementation is [openvinotoolkit/anomalib](https://github.com/openvinotoolkit/anomalib).

<p align="center">
<img src="https://github.com/wang-xinyu/tensorrtx/assets/15235574/061c90a7-fe59-48e0-a8d0-6bddc4296cf1">
</p>

# Test Environment

GTX3080 / Windows10 22H2 / cuda11.8 / cudnn8.9.7 / TensorRT8.5.3 / OpenCV4.6

# How to Run

1. training to generate weight files (`efficientAD_[category].pt`)

   ```
   // Please refer to Anomalib's tutorial for details:
   // https://github.com/openvinotoolkit/anomalib?tab=readme-ov-file#-training
   ```

2. generate `.wts` from pytorch with `.pt`

   ```
   cd ./datas/models/
   // copy your `.pt` file to the current directory.
   python gen_wts.py
   // a file `efficientAD_[category].wts` will be generated.
   ```

3. build and run

   ```
   mkdir build
   cd build
   cmake ..
   make
   sudo ./EfficientAD-M -s [.wts] // serialize model to plan file
   sudo ./EfficientAD-M -d [.engine] [image folder] // deserialize and run inference, the images in [image folder] will be processed
   ```

# Latency

average cost of doInference(in `efficientad_detect.cpp`) from second time with batch=1 under the windows environment above

|               | FP32 |
| :-----------: | :--: |
| EfficientAD-M | 12ms |
