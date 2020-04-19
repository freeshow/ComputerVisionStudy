**《Very Deep Convolutional Networks for Large-Scale Image Recognition》**

- arXiv：[[1409.1556\] Very Deep Convolutional Networks for Large-Scale Image Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.1556)
- intro：ICLR 2015
- homepage：[Visual Geometry Group Home Page](https://link.zhihu.com/?target=http%3A//www.robots.ox.ac.uk/~vgg/research/very_deep/)

## **1、VGG简介**

自Alexnet于2012年的ILSVRC大赛中奋勇夺魁后，ConvNet就不可抑制地大发展了起来。2014年新的一届ILSVRC大赛中Googlenet与VGG的身影分外亮眼。Googlenet相对VGG而言在网络结构上有了更新的突破，不过其复杂度也大大增加了。VGG相对Googlenet虽然精度略逊些，但其整体网络框架还是延续了Alexnet及更早的Lenet等的一贯思路，此外还更深入的探讨了ConvNet深度对模型性能可能的影响。由于其整个网络结构的简单、强大，VGG16/VGG19曾一度广泛被用作各种检测网络框架像Faster-RCNN/SSD等的主干特征提取网络，直到Resnet提出之后，它才渐渐完成了其历史使命，退居二线了。。不过至今仍有许多计算机视觉领域内的任务会考虑VGG的网络设计来构建其新的应用网络模型。

牛津大学VGG(Visual Geometry Group)组在2014年ILSVRC提出的模型被称作VGG模型 。该模型相比以往模型进一步加宽和加深了网络结构，它的核心是五组卷积操作，每两组之间做Max-Pooling空间降维。同一组内采用多次连续的3X3卷积，卷积核的数目由较浅组的64增多到最深组的512，同一组内的卷积核数目是一样的。卷积之后接两层全连接层，之后是分类层。由于每组内卷积层的不同，有11、13、16、19层这几种模型，下图展示一个16层的网络结构。VGG模型结构相对简洁，提出之后也有很多文章基于此模型进行研究，如在ImageNet上首次公开超过人眼识别的模型就是借鉴VGG模型的结构。

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414151535.png)

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414145643.png)

## 2、VGG原理

VGG 相比 AlexNet 的一个改进是**采用连续的几个3x3的卷积核代替较大卷积核（11x11，7x7，5x5）**。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。

简单来说，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

比如，3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），其参数总量为 3x(9xC^2) ，如果直接使用7x7卷积核，其参数总量为 49xC^2 ，这里 C 指的是输入和输出的通道数。很明显，27xC^2小于49xC^2，即减少了参数；而且3x3卷积核有利于更好地保持图像性质。

作用在于：

1. 增加了两层非线性层 ReLU，

2. 减少了网络的参数

`为什么使用2个3x3卷积核可以来代替5*5卷积核：`

5x5卷积看做一个小的全连接网络在5x5区域滑动，我们可以先用一个3x3的卷积滤波器卷积，然后再用一个全连接层连接这个3x3卷积输出，这个全连接层我们也可以看做一个3x3卷积层。这样我们就可以用两个3x3卷积级联（叠加）起来代替一个 5x5卷积。

具体如下图所示：

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414153310.png)

至于为什么使用3个3x3卷积核可以来代替7*7卷积核，推导过程与上述类似，大家可以自行绘图理解。

`参数优势：`

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414153523.png)

由于参数个数仅与卷积核大小有关，所以3*3级联卷积核占优势。

## **3、VGG网络结构**

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/vgg16.png)

下面是VGG网络的结构（VGG16和VGG19都在）：

![VGG网络](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200414155459.png)

可以从图中看出，从A到最后的E，他们增加的是每一个卷积组中的卷积层数，最后D，E是我们常见的VGG-16，VGG-19模型，C中作者说明，在引入1*1是考虑做线性变换（这里channel一致， 不做降维），后面在最终数据的分析上来看C相对于B确实有一定程度的提升，但不如D。

- VGG16包含了16个隐藏层（13个卷积层和3个全连接层），如上图中的D列所示

- VGG19包含了19个隐藏层（16个卷积层和3个全连接层），如上图中的E列所示

VGG网络的结构非常一致，从头到尾全部使用的是3x3的卷积和2x2的max pooling。

如果你想看到更加形象化的VGG网络，可以使用[经典卷积神经网络（CNN）结构可视化工具](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/gktWxh1p2rR2Jz-A7rs_UQ)来查看高清无码的[VGG网络](https://link.zhihu.com/?target=https%3A//dgschwend.github.io/netscope/%23/preset/vgg-16)。

## **4、VGG优缺点**

**VGG优点**

- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。
- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：
- 验证了通过不断加深网络结构可以提升性能。

**VGG缺点**

- VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG可是有3个全连接层啊！

PS：有的文章称：发现这些全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量。

> 注：很多 pretrained 的方法就是使用 VGG 的 model（主要是16和19），VGG 相对其他的方法，参数空间很大，最终的 model 有500多m，AlexNet只有200m，GoogLeNet更少，所以train一个vgg模型通常要花费更长的时间，所幸有公开的 pretrained model 让我们很方便的使用。

## 5、VGG Tensorflow2.0实现

GitHub地址：https://github.com/freeshow/ComputerVisionStudy/blob/master/VGG/VGG16.ipynb

## 6、相关链接

- https://zhuanlan.zhihu.com/p/41423739