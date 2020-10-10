# SSD MobileNet v2

MobileNetV2 architecture from
     "MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>.

For the Pytorch implementation, you can refer to [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)

### How to run
1. Generate weights from  using `gen_wts.py` script
```
git clone https://github.com/qfgaohao/pytorch-ssd.git
cd pytorch-ssd
cp {tensorrtx}/ssd/gen_wts.py .
python gen_wts.py
```
2. Put ssdmobilenet.wts into tensorrtx/ssd
```
cp {pytorch-ssd}/models/ssdmobilenet.wts {tensorrtx}/ssd/ssd_mobilenet.wts
```
3. Build and run
```
cd {tensorrtx}/ssd
mkdir build
cd build
cmake ..
make
sudo ./ssd-mobilenetv2 -s   // serialize model to plan file i.e. 'ssd_mobilenet.engine'
sudo ./ssd-mobilenetv2 -d   // deserialize plan file and run inference
```
4. Check if the output is same as {pytorch-ssd}
