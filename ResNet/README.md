论文地址：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 一、引言

深度残差网络（Deep residual network, ResNet）的提出是CNN图像史上的一件里程碑事件，让我们先看一下ResNet在ILSVRC和COCO 2015上的战绩：

![图1 ResNet在ILSVRC和COCO 2015上的战绩](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420094523.png)

ResNet取得了5项第一，并又一次刷新了CNN模型在ImageNet上的历史：

![图2 ImageNet分类Top-5误差](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420094603.png)

ResNet的作者[何凯明](https://link.zhihu.com/?target=http%3A//kaiminghe.com/)也因此摘得CVPR2016最佳论文奖，当然何博士的成就远不止于此，感兴趣的可以去搜一下他后来的辉煌战绩。那么ResNet为什么会有如此优异的表现呢？其实ResNet是解决了深度CNN模型难训练的问题，从图2中可以看到14年的VGG才19层，而15年的ResNet多达152层，这在网络深度完全不是一个量级上，所以如果是第一眼看这个图的话，肯定会觉得ResNet是靠深度取胜。事实当然是这样，但是ResNet还有架构上的trick，这才使得网络的深度发挥出作用，这个trick就是残差学习（Residual learning）。下面详细讲述ResNet的理论及实现。

## 二、深度网络的退化问题

从经验来看，网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时理论上可以取得更好的结果，从图2中也可以看出网络越深而效果越好的一个实践证据。但是更深的网络其性能一定会更好吗？实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。这个现象可以在图3中直观看出来：56层的网络比20层网络效果还要差。这不会是过拟合问题，因为56层网络的训练误差同样高。我们知道深层网络存在着梯度消失或者爆炸的问题，这使得深度学习模型很难训练。但是现在已经存在一些技术手段如BatchNorm来缓解这个问题。因此，出现深度网络的退化问题是非常令人诧异的。

![图3 20层与56层网络在CIFAR-10上的误差](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420092113.png)

从上面两个图可以看出，在网络很深的时候（56层相比20层），模型效果却越来越差了（误差率越高），并不是网络越深越好。
通过实验可以发现：**随着网络层级的不断增加，模型精度不断得到提升，而当网络层级增加到一定的数目以后，训练精度和测试精度迅速下降，这说明当网络变得很深以后，深度网络就变得更加难以训练了。**

**训练集上的性能下降，可以排除过拟合，BN层的引入也基本解决了plain net的梯度消失和梯度爆炸问题。**如果不是过拟合以及梯度消失导致的，那原因是什么？

按道理，给网络叠加更多层，浅层网络的解空间是包含在深层网络的解空间中的，深层网络的解空间至少存在不差于浅层网络的解，因为只需将增加的层变成恒等映射，其他层的权重原封不动copy浅层网络，就可以获得与浅层网络同样的性能。**更好的解明明存在，为什么找不到？找到的反而是更差的解？**

显然，这是个优化问题，反映出结构相似的模型，其优化难度是不一样的，且难度的增长并不是线性的，越深的模型越难以优化。

有两种解决思路，**一种是调整求解方法，比如更好的初始化、更好的梯度下降算法等；另一种是调整模型结构，让模型更易于优化——改变模型结构实际上是改变了error surface的形态。**

ResNet的作者从后者入手，探求更好的模型结构。将堆叠的几层layer称之为一个block，对于某个block，其可以拟合的函数为$F(x)$，如果期望的潜在映射为$H(x)$，**与其让 $F(x)$  直接学习潜在的映射，不如去学习残差 $H(x)−x$，即 $F(x):=H(x)−x$，这样原本的前向路径上就变成了 $F(x)+x$ ，用 $F(x)+x$来拟合$H(x)$。**作者认为这样可能更易于优化，因为**相比于让$F(x)$学习成恒等映射，让$F(x)$学习成0要更加容易——后者通过L2正则就可以轻松实现。**这样，对于冗余的block，只需$F(x)→0$就可以得到恒等映射，性能不减。

下面的问题就变成了 $F(x)+x$ 该怎么设计了。

## 三、Residual Block的设计

前面描述了一个实验结果现象，在不断加神经网络的深度时，模型准确率会先上升然后达到饱和，再持续增加深度时则会导致准确率下降，示意图如下：

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420092900.png)

> 那么我们作这样一个假设：假设现有一个比较浅的网络（Shallow Net）已达到了饱和的准确率，这时在它后面再加上几个恒等映射层（Identity mapping，也即y=x，输出等于输入），这样就增加了网络的深度，并且起码误差不会增加，也即更深的网络不应该带来训练集上误差的上升。而这里提到的使用恒等映射直接将前一层输出传到后面的思想，便是著名深度残差网络ResNet的灵感来源。

这个有趣的假设让何博士灵感爆发，他提出了残差学习来解决退化问题。对于一个堆积层结构（几层堆积而成）当输入为 $x$ 时其学习到的特征记为  $H(x)$，现在我们希望其可以学习到`残差` $F(x)=H(x)-x$ ，这样其实原始的学习特征是$F(x)+x$  。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为 0 时，此时堆积层仅仅做了**恒等映射**，至少网络性能不会下降，实际上残差不会为 0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。残差学习的结构如图4所示。这有点类似与电路中的“短路”，所以是一种**短路连接（shortcut connection）**。

![图4 残差学习单元](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420101359.png)

$F(x)+x$ 构成的block称之为**Residual Block**，即**残差块**，如上图所示，多个相似的Residual Block串联构成ResNet。

一个残差块有2条路径 $F(x)$ 和 $x$，$F(x)$ 路径拟合残差，不妨称之为**残差路径**，$x$ 路径为 identity mapping恒等映射，称之为”**shortcut**”。**图中的⊕为element-wise addition，要求参与运算的 $F(x)$ 和$x$ 的尺寸要相同**。所以，随之而来的问题是：

- 残差路径如何设计？
- shortcut路径如何设计？
- Residual Block之间怎么连接？

在原论文中，残差路径可以大致分成2种，一种有**bottleneck**结构，即下图右中的1×1卷积层，用于先降维再升维，主要出于**降低计算复杂度的现实考虑**，称之为“**bottleneck block**”，另一种没有bottleneck结构，如下图左所示，称之为“**basic block**”。basic block由2个3×3卷积层构成，bottleneck block由1×1

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420111839.png)

**bottleneck block**中第一个1x1的卷积把256维channel降到64维，然后在最后通过1x1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632，而是**basic block**中两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648，差了16.94倍。

> 对于常规ResNet，可以用于34层或者更少的网络中，对于Bottleneck Design的ResNet通常用于更深的如101这样的网络中，目的是减少计算和参数量（**实用目的**）。

shortcut路径大致也可以分成2种，取决于残差路径是否改变了feature map数量和尺寸，一种是将输入x原封不动地输出，**另一种则需要经过1×1卷积来升维 or/and 降采样**，主要作用是**将输出与F(x)路径的输出保持shape一致**，对网络性能的提升并不明显，两种结构如下图所示。

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420112149.png)



至于Residual Block之间的衔接，在原论文中，$F(x)+x$ 经过ReLU后直接作为下一个block的输入x。

## 四、ResNet 网络结构

ResNet为多个Residual Block的串联，下面直观看一下ResNet-34与34-layer plain net和VGG的对比，以及堆叠不同数量Residual Block得到的不同ResNet。

![ResNet网络结构图](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420113832.png)

![不同深度的ResNet](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420114226.png)

ResNet的设计有如下特点：

- 与plain net相比，ResNet多了很多“旁路”，即shortcut路径，其首尾圈出的layers构成一个Residual Block；
- ResNet中，所有的Residual Block都没有pooling层，**降采样是通过conv的stride实现的**；
- 分别在conv3_1、conv4_1和conv5_1 Residual Block，降采样1倍，同时feature map数量增加1倍，如图中虚线划定的block；
- **通过Average Pooling得到最终的特征**，而不是通过全连接层；
- 每个卷积层之后都紧接着BatchNorm layer，为了简化，图中并没有标出；

> **ResNet结构非常容易修改和扩展，通过调整block内的channel数量以及堆叠的block数量，就可以很容易地调整网络的宽度和深度，来得到不同表达能力的网络，而不用过多地担心网络的“退化”问题，只要训练数据足够，逐步加深网络，就可以获得更好的性能表现。**

下面为网络的性能对比：

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420114816.png)

## 五、error surface对比

上面的实验说明，不断地增加ResNet的深度，甚至增加到1000层以上，也没有发生“退化”，可见Residual Block的有效性。ResNet的动机在于**认为拟合残差比直接拟合潜在映射更容易优化**，下面通过绘制error surface直观感受一下shortcut路径的作用，图片截自[Loss Visualization](http://www.telesens.co/loss-landscape-viz/viewer.html)。

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420115106.png)

可以发现：

- ResNet-20（no short）浅层plain net的error surface还没有很复杂，优化也会很困难，但是增加到56层后复杂程度极度上升。**对于plain net，随着深度增加，error surface 迅速“恶化”**；
- 引入shortcut后，**error suface变得平滑很多，梯度的可预测性变得更好，显然更容易优化**；

## 六、Residual Block的分析与改进

论文[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)进一步研究ResNet，通过ResNet反向传播的理论分析以及调整Residual Block的结构，得到了新的结构，如下

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200420115329.png)

**注意，这里的视角与之前不同，这里将shortcut路径视为主干路径，将残差路径视为旁路。**

新提出的Residual Block结构，具有更强的泛化能力，能更好地避免“退化”，堆叠大于1000层后，性能仍在变好。具体的变化在于

- **通过保持shortcut路径的“纯净”，可以让信息在前向传播和反向传播中平滑传递，这点十分重要。**为此，如无必要，不引入1×1卷积等操作，同时将上图灰色路径上的ReLU移到了$F(x)$路径上。
- 在残差路径上，**将BN和ReLU统一放在weight前作为pre-activation**，获得了“Ease of optimization”以及“Reducing overfitting”的效果。

## 七、ResNet18 Tensorflow2.0 实现

GitHub地址：https://github.com/freeshow/ComputerVisionStudy

## 八、参考链接

- https://my.oschina.net/u/876354/blog/1622896
- https://zhuanlan.zhihu.com/p/31852747
- https://www.cnblogs.com/shine-lee/p/12363488.html



![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/公众号宣传码.png)