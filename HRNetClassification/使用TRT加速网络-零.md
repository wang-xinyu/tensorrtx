# 使用TRT加速网络-零

本次教程以HRNet分类器（HRNet-W18-C-Small-v2）为例子

code：https://github.com/HRNet/HRNet-Image-Classification

paper：https://arxiv.org/abs/1908.07919

## 1 论文网络的基本了解

无论是仅仅使用网络还是要对网络改进，首先都要对网络有一定了解。对于这种比较火的网络，网上大批详解博客，可以多去阅读，加上论文，来对网络理解。

HRNet分类器网络看起来很简单，如下图

![682463-20200104221712824-157549407](https://user-images.githubusercontent.com/20653176/93749152-ff957680-fc2b-11ea-883c-79046e41ace8.png)

从网络中可看到基本组件很简单：卷积和upsmple。【这里就表明网络TRT加速时不会有plugin的需求。】

参考博客：

1. https://www.cnblogs.com/darkknightzh/p/12150637.html
2. https://zhuanlan.zhihu.com/p/143385915
3. https://blog.csdn.net/weixin_37993251/article/details/88043650
4. https://blog.csdn.net/weixin_38715903/article/details/101629781?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=dis

## 2 pytorch代码跑通

跑通demo是很重要的一步。跑通后就可以一步一步跟进，看到底走了哪些层，这样心里就会有一个基本框架；然后可以生成wts文件；同时也可以生成onnx文件。

上述的**参考博客4**中对代码有详细介绍，可以详细分析下。

建议：**对于运行环境，建议使用anaconda的conda create 创建虚拟环境，这样没有一系列环境问题。**

```python
conda create -n xx python=3.7   # 创建环境
activate xx    # 激活
pip install xxxx  # 安装包
deactivate xx  # 推出环境
```

在生成wts文件时，没有必须每次都是去配置`gen_wts.py`，主要是读取模型，保存模型参数。只要demo文件跑通就可以随时保存为wts。

## 3 pytorch代码debug

这一步骤单独拉出来是因为在debug的过程中，要关注经过哪些层，预处理有哪些，后处理有哪些。另外在后面搭建TRT网络时，还要根据debug过程在中的一些信息来调试trt网络。

## 4 网络的可视化

将pytorch模型保存为onnx，可有可无。但是建议如果可以保存，就使用onnx来可视化网络。这样对网络架构一级每层的输入输出就会非常明了。

如果无法保存onnx，搭建网络时，要根据wts来分析，比较麻烦。

另外强烈建议：**无论是否保存了onnx，都要手动在纸上将网络在画一遍，，并且将每层的输出维度标注下来，这样搭建层比较多的网络时，不会晕，并且在debugTRT网络时可以有效定位错误。**

在手动画网络图时，可以给每个节点“标号”，利用该“标号”在搭建TRT网络时，可以很清楚知道 **“哪个节点输入，经过某种操作，输出哪个节点。”**

在onnx图中看到几个层一定要心里有数：

比如下面红线框出的一大块实际上就是upsample层

![](imgs/93747936-0ae7a280-fc2a-11ea-86c1-9f72622402b9.png))

下面的为FC层：

![image-20200918141448071](https://user-images.githubusercontent.com/20653176/93749177-0de39280-fc2c-11ea-8a20-b8ab0b3b940f.png)

Conv+BN+Relu层

![image-20200918141632723](https://user-images.githubusercontent.com/20653176/93749201-189e2780-fc2c-11ea-9aad-0ac7723575c4.png)

ResBlock层

![image-20200918141709487](https://user-images.githubusercontent.com/20653176/93749220-2358bc80-fc2c-11ea-998a-0892755dfbc0.png)

单击节点。会有详细信息，这些信息使搭建网络变得方便。

![image-20200918141931327](https://user-images.githubusercontent.com/20653176/93749222-2489e980-fc2c-11ea-9025-c5d367efd7f9.png)



如果无法导出onnx：

搭建网络时需要从wts中查看层名，各个卷积层信息需要从代码中分析。

![image_f](https://user-images.githubusercontent.com/20653176/93750398-fd341c00-fc2d-11ea-9077-ee749b6aef41.png)

![image-20200918142959711](https://user-images.githubusercontent.com/20653176/93749484-8fd3bb80-fc2c-11ea-951d-3c1f403e521a.png)

## 5 TRT搭建网络

搭建网络时就按照onnx图一层一层搭建。

几点建议：

1 要不断去查API的使用 https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html

2 利用已有的模块，不要重复造轮子

3 各个层名使用onnx 的id，这样在搭建网络时不会晕。，根据onnx的结点信息，各层之间的连接也不会出错。



## 6 TRT网络debug

搭建网络过程肯定会出错，debug是必要的手段：

1 打印每层的维度

```c++
Dims dim = id_1083->getOutput(0)->getDimensions();
std::cout << dim[0] << " " << dim[1] << " " << dim[2] << " " << dim[3] << std::endl; 
```

**一般如果出现生成engine就失败的情况，就从createEngine的第一句开始调试，并且随时关注窗口输出，如果在某一层出现大量提示信息，那么该层就会有问题，就将该层的输入tensor维度和输出tensor维度信息都打印出来，看输出的维度是否正常。**

2 打印输出

TRT是先构建网络，然后再enqueue时才能得到各层的输出信息，因此若想对比每一层的输出，需要将该层设置为output层

```c++
out->getOutput(0)->setName(OUTPUT_BLOB_NAME);  // out可替换为任意一层
network->markOutput(*out->getOutput(0));
```

3 关注输入层data

数据层的debug无需第2步的做法，直接可以查看预处理后的结果。在debug

## 7 TRT代码整理

这里就是将TRT搭建的网络，能封装函数，就封装为函数模块，增加代码可读性。
