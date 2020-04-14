AlexNet模型来源于论文-[ImageNet Classification with Deep Convolutional Neural Networks](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/~fritz/absps/imagenet.pdf)，作者Alex Krizhevsky，Ilya Sutskever，Geoffrey E.Hinton.

`转载自：` https://zhuanlan.zhihu.com/p/42914388

## **1. 前言**

在2010年的 ImageNet LSVRC-2010上，AlexNet 在给包含有1000种类别的共120万张高分辨率图片的分类任务中，在测试集上的top-1和top-5错误率为37.5%和17.0%（**top-5 错误率：即对一张图像预测5个类别，只要有一个和人工标注类别相同就算对，否则算错。同理top-1对一张图像只预测1个类别**），在 ImageNet LSVRC-2012 的比赛中，取得了top-5错误率为15.3%的成绩。AlexNet 有6亿个参数和650,000个神经元，包含5个卷积层，有些层后面跟了max-pooling层，3个全连接层，为了减少过拟合，在全连接层使用了dropout，下面进行更加详细的介绍。

## **2. 数据集**

数据来源于[ImageNet](https://link.zhihu.com/?target=http%3A//www.image-net.org/)，训练集包含120万张图片，验证集包含5万张图片，测试集包含15万张图片，这些图片分为了1000个类别，并且有多种不同的分辨率，但是AlexNet的输入要求是固定的分辨率，为了解决这个问题，Alex的团队采用低采样率把每张图片的分辨率降为256×256，具体方法就是给定一张矩形图像，首先重新缩放图像，使得较短边的长度为256，然后从结果图像的中心裁剪出256×256大小的图片。

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414093505.png)

## **3. 网络结构**

先看几张网络的结构图：

![论文原文中的图](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414093706.png)

该结构分为上下两个部分，每部分含有五层卷积层和三层全连接层，之所以分为两部分是为了方便在两片GPU上进行训练，只在第三层卷积层和全连接层处上下两部分可以交互。由于上下两部分完全一致，分析时一般取一部分即可。

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/alexnet_meitu_1.png)

![细化的结构图](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414094257.png)

下面对网络中的一些细节进行介绍

### **3.1 非线性ReLU函数**

在当时，标准的神经元激活函数是tanh()函数，即

这种饱和的非线性函数在梯度下降的时候要比非饱和的非线性函数慢得多，因此，在AlexNet中使用ReLU函数作为激活函数

下面这种图展示了在一个4层的卷积网络中使用ReLU函数在CIFAR-10数据集上达到25%的训练错误率要比在相同网络相同条件下使用tanh函数快6倍

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414094707.png)

### **3.2 多GPU训练**

AlexNet采用两路GTX 580 3G并行训练，并行训练的示意图如下图所示

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414094738.png)

值得注意的是，这种训练方法使top-1和top-5错误率和使用一个GPU训练一半的kernels相比分别降低了1.7%和1.2%，

### **3.3 局部响应归一化（Local Response Normalization,LRN）**

ReLU函数不像tanh和sigmoid一样有一个有限的值域区间，所以在ReLU之后需要进行归一化处理，LRN的思想来源于神经生物学中一个叫做“侧抑制”的概念，指的是被激活的神经元抑制周围的神经元。计算公式为:

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/lrn_meitu_1.png)

>但似乎，在后来的设计中，这一层已经被其它种的Regularization技术，如drop out, batch normalization取代了。
>
>后面VGGNet等证明LRN并没有想象中那么好的效果，现在流行的规范化选择是BN。

### **3.4 重叠池化（Overlapping Pooling）**

普通的池化操作不带重叠，其stride就是kernel的大小。最大池化就是寻找最大值，即最能代表数据显著性特征的值，平均池化就是计算框内的平均值，即为数据层面上的均衡值。

![普通池化操作](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414100916.png)

覆盖的池化操作就是指相邻池化窗口之间有重叠部分，在池化的过程中会存在重叠的部分，即stride<kernel_size

作者实验表明重叠的池化层更不容易过拟合。在AlexNet中 ，这样的设定使他们的top-1和top-5错误率分别降低了0.4%和0.3%（和使用不重叠的池化 相比）。

### **3.5 总体结构**

网络的最后一层（Full8）的输出喂给了一个包含1000个单元的softmax层，用来对1000个标签进行预测。响应归一化层（Response-normalization layers）跟在第1和第2卷积层后面，Max-pooling层跟在Response-normalization层和第5卷积层后面，ReLU激活函数应用与所有卷积层和全连接层输出后。

## **4. 减少过拟合**

下面介绍AlexNet中使用的缓解过拟合的两个主要方法。

### **4.1 Data Augmentation（数据增量）**

早期最常见的针对图像数据减少过拟合的方法就是人工地增大数据集，AlexNet中使用了两种增大数据量的方法

1、镜像反射和随机剪裁

先对图像做镜像反射，就像下图这样：

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414101426.png)

然后在原图和镜像反射的图（256×256）中随机抽取227×227的块，像这样：

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414101451.png)

通过这种方法，使得训练集的大小增大了2048倍，尽管由此产生的训练样例会产生高度的相互依赖。但是不使用这种方法又会导致严重的过拟合，迫使我们使用更小的网络。在测试的时候，AlexNet会抽取测试样本及其镜像反射图各5块（总共10块，四个角和中心位置）来进行预测，预测结果是这10个块的softmax块的平均值。

> 在[OverFeat](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.6229)这篇文章中，作者指出，这样裁剪测试的方法会忽略掉图片中的很多区域，并且从计算角度讲，裁剪窗口重叠存在很多冗余的计算，另外，裁剪窗口只有单一的尺寸，这可能不是ConvNet的最佳置信度的尺寸。

2、改变训练样本RGB通道的强度值

Alex团队在整个训练集中对图片的RGB像素值集执行PCA（[主成分分析，Principal Components Analysis](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/zh-hans/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)）。对于每一张训练图片，他们增加了多个找到的主成分，它们的大小比例是相应的特征值乘以一个随机值（来自均值为0，标准差为0.1的高斯分布）。

### **4.2 Dropout**

关于Dropout的详细介绍参考-[《深度学习》课程总结-深度学习的实用层面](https://zhuanlan.zhihu.com/p/40423613)，在AlexNet中设置的失活概率为0.5，在测试的时候，再使用所用的神经元但是要给它们的输出都乘以0.5。

## **5. 更多细节**

AlexNet使用随机梯度下降算法，batch大小是128，动量衰减参数设置为0.9，权重衰减参数为0.0005，这里的权重衰减不仅仅是一个正规化器，同时它减少了模型的训练误差。

另外，在AlexNet中，所以层的权重 初始化为服从0均值，标准差为0.001的高斯分布，第2、4、5卷积层以及全连接层的偏置量 初始化为1，这样做的好处是它通过给ReLU函数一个正激励从而加速早期学习的速度。其他层的偏置量初始化为0.

## 6. AlexNet Tensorflow2.0实现

`GitHub仓库地址：` https://github.com/freeshow/ComputerVisionStudy.git

```python
# tensorflow2.0
import tensorflow as tf
from tensorflow import keras

class LRN(keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius=2
        self.bias=1
        self.alpha=1e-4
        self.beta=0.75
    def call(self,x):
        return tf.nn.lrn(x,depth_radius=self.depth_radius,
                         bias=self.bias,alpha=self.alpha,
                         beta=self.beta)
    
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=96,
                              kernel_size=(11,11),
                              strides=4,
                              activation='relu',
                              padding='same',
                              input_shape=(227,227,3)))
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
model.add(LRN())
model.add(keras.layers.Conv2D(filters=256,
                              kernel_size=(5,5),
                              strides=1,
                              activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
model.add(LRN())
model.add(keras.layers.Conv2D(filters=384,
                              kernel_size=(3,3),
                              strides=1,
                              activation='relu',
                              padding='same'))

model.add(keras.layers.Conv2D(filters=384,
                              kernel_size=(3,3),
                              strides=1,
                              activation='relu',
                              padding='same'))
model.add(keras.layers.Conv2D(filters=256,
                              kernel_size=(3,3),
                              strides=1,
                              activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1000,activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics = ["accuracy"])
```

![欢迎关注微信公众号：大数据AI](http://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200302/224029974.png)