# 使用 TRT 加速网络-零

本次教程以 HRNet 分类器（HRNet-W18-C-Small-v2）为例子

code：https://github.com/HRNet/HRNet-Image-Classification

paper：https://arxiv.org/abs/1908.07919

## 1 论文网络的基本了解

无论是仅仅使用网络还是要对网络改进，首先都要对网络有一定了解。对于这种比较火的网络，网上大批详解博客，可以多去阅读，加上论文，来对网络理解。

HRNet 分类器网络看起来很简单，如下图

![682463-20200104221712824-157549407](https://user-images.githubusercontent.com/20653176/93749152-ff957680-fc2b-11ea-883c-79046e41ace8.png)

从网络中可看到基本组件很简单：卷积和 upsmple。【这里就表明网络 TRT 加速时不会有 plugin 的需求。】

参考博客：

1. https://www.cnblogs.com/darkknightzh/p/12150637.html
2. https://zhuanlan.zhihu.com/p/143385915
3. https://blog.csdn.net/weixin_37993251/article/details/88043650
4. https://blog.csdn.net/weixin_38715903/article/details/101629781?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=dis

## 2 pytorch 代码跑通

跑通 demo 是很重要的一步。跑通后就可以一步一步跟进，看到底走了哪些层，这样心里就会有一个基本框架；然后可以生成 wts 文件；同时也可以生成 onnx 文件。

上述的**参考博客 4**中对代码有详细介绍，可以详细分析下。

建议：**对于运行环境，建议使用 anaconda 的 conda create 创建虚拟环境，这样没有一系列环境问题。**

```python
conda create -n xx python=3.7   # 创建环境
activate xx    # 激活
pip install xxxx  # 安装包
deactivate xx  # 推出环境
```

在生成 wts 文件时，没有必须每次都是去配置`gen_wts.py`，主要是读取模型，保存模型参数。只要 demo 文件跑通就可以随时保存为 wts。

## 3 pytorch 代码 debug

这一步骤单独拉出来是因为在 debug 的过程中，要关注经过哪些层，预处理有哪些，后处理有哪些。另外在后面搭建 TRT 网络时，还要根据 debug 过程在中的一些信息来调试 trt 网络。

## 4 网络的可视化

将 pytorch 模型保存为 onnx，可有可无。但是建议如果可以保存，就使用 onnx 来可视化网络。这样对网络架构一级每层的输入输出就会非常明了。

如果无法保存 onnx，搭建网络时，要根据 wts 来分析，比较麻烦。

另外强烈建议：**无论是否保存了 onnx，都要手动在纸上将网络在画一遍，，并且将每层的输出维度标注下来，这样搭建层比较多的网络时，不会晕，并且在 debugTRT 网络时可以有效定位错误。**

在手动画网络图时，可以给每个节点“标号”，利用该“标号”在搭建 TRT 网络时，可以很清楚知道 **“哪个节点输入，经过某种操作，输出哪个节点。”**

在 onnx 图中看到几个层一定要心里有数：

比如下面红线框出的一大块实际上就是 upsample 层

![](imgs/93747936-0ae7a280-fc2a-11ea-86c1-9f72622402b9.png))

下面的为 FC 层：

![image-20200918141448071](https://user-images.githubusercontent.com/20653176/93749177-0de39280-fc2c-11ea-8a20-b8ab0b3b940f.png)

Conv+BN+Relu 层

![image-20200918141632723](https://user-images.githubusercontent.com/20653176/93749201-189e2780-fc2c-11ea-9aad-0ac7723575c4.png)

ResBlock 层

![image-20200918141709487](https://user-images.githubusercontent.com/20653176/93749220-2358bc80-fc2c-11ea-998a-0892755dfbc0.png)

单击节点。会有详细信息，这些信息使搭建网络变得方便。

![image-20200918141931327](https://user-images.githubusercontent.com/20653176/93749222-2489e980-fc2c-11ea-9025-c5d367efd7f9.png)

如果无法导出 onnx：

搭建网络时需要从 wts 中查看层名，各个卷积层信息需要从代码中分析。

![image_f](https://user-images.githubusercontent.com/20653176/93750398-fd341c00-fc2d-11ea-9077-ee749b6aef41.png)

![image-20200918142959711](https://user-images.githubusercontent.com/20653176/93749484-8fd3bb80-fc2c-11ea-951d-3c1f403e521a.png)

## 5 TRT 搭建网络

搭建网络时就按照 onnx 图一层一层搭建。

几点建议：

1 要不断去查 API 的使用 https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html

2 利用已有的模块，不要重复造轮子

3 各个层名使用 onnx 的 id，这样在搭建网络时不会晕。，根据 onnx 的结点信息，各层之间的连接也不会出错。

## 6 TRT 网络 debug

搭建网络过程肯定会出错，debug 是必要的手段：

1 打印每层的维度

```c++
Dims dim = id_1083->getOutput(0)->getDimensions();
std::cout << dim[0] << " " << dim[1] << " " << dim[2] << " " << dim[3] << std::endl;
```

**一般如果出现生成 engine 就失败的情况，就从 createEngine 的第一句开始调试，并且随时关注窗口输出，如果在某一层出现大量提示信息，那么该层就会有问题，就将该层的输入 tensor 维度和输出 tensor 维度信息都打印出来，看输出的维度是否正常。**

2 打印输出

TRT 是先构建网络，然后再 enqueue 时才能得到各层的输出信息，因此若想对比每一层的输出，需要将该层设置为 output 层

```c++
out->getOutput(0)->setName(OUTPUT_BLOB_NAME);  // out可替换为任意一层
network->markOutput(*out->getOutput(0));
```

3 关注输入层 data

数据层的 debug 无需第 2 步的做法，直接可以查看预处理后的结果。在 debug

## 7 TRT 代码整理

这里就是将 TRT 搭建的网络，能封装函数，就封装为函数模块，增加代码可读性。
