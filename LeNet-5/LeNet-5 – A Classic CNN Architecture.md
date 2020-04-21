

LeNet 诞生于 1994 年，是最早的卷积神经网络之一，并且推动了深度学习领域的发展。自从 1988 年开始，在许多次成功的迭代后，这项由 Yann LeCun 完成的开拓性成果被命名为 LeNet5。

LeNet-5 出自论文 [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)，是一种用于手写体字符识别的非常高效的卷积神经网络。Lenet-5 是 Yann LeCun 提出的，对 MNIST 数据集的分识别准确度可达 **99.2%**。

## 1、LeNet-5 Architecture

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413104002.png)

The LeNet-5 architecture consists of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier.

### 1.1 First Layer

The input for LeNet-5 is a 32×32 grayscale image which passes through the first convolutional layer with 6 feature maps or filters having size 5×5 and a stride of one. The image dimensions changes from 32x32x1 to 28x28x6.

![C1: Convolutional Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413114454.png)

### 1.2 Second Layer

Then the LeNet-5 applies average pooling layer or sub-sampling layer with a filter size 2×2 and a stride of two. The resulting image dimensions will be reduced to 14x14x6.

![S2: Average Pooling Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413114652.png)

### 1.3 Third Layer

Next, there is a second convolutional layer with 16 feature maps having size 5×5 and a stride of 1. In this layer, only 10 out of 16 feature maps are connected to 6 feature maps of the previous layer as shown below.

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413121754.png)

The main reason is to break the symmetry in the network and keeps the number of connections within reasonable bounds. That’s why the number of training parameters in this layers are 1516 instead of 2400 and similarly, the number of connections are 151600 instead of 240000.

![C3: Convolutional Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413122035.png)

### 1.4 Fourth Layer

The fourth layer (S4) is again an average pooling layer with filter size 2×2 and a stride of 2. This layer is the same as the second layer (S2) except it has 16 feature maps so the output will be reduced to 5x5x16.

![S4: Average Pooling Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413122418.png)

### 1.5 Fifth Layer

The fifth layer (C5) is a fully connected convolutional layer with 120 feature maps each of size 1×1. Each of the 120 units in C5 is connected to all the 400 nodes (5x5x16) in the fourth layer S4.

![C5: Fully Connected Convolutional Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413122556.png)

### 1.6 Sixth Layer

The sixth layer is a fully connected layer (F6) with 84 units.

![F6: Fully Connected Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413122702.png)

### 1.7 Output Layer

Finally, there is a fully connected softmax output layer ŷ with 10 possible values corresponding to the digits from 0 to 9.

![Fully Connected Output Layer](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413122747.png)

## 2、Summary of LeNet-5 Architecture

![LeNet-5 Architecture Summarized Table](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413122833.png)

![LeNet-5 Architecture](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/20200413123039.png)

## 3、Implementation of LeNet-5 Using Tensorflow2.0

`GitHub仓库地址：` https://github.com/freeshow/ComputerVisionStudy.git

### 3.1 导入相关包

```python
import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from tensorflow.keras.callbacks import TensorBoard
```

### 3.2 加载数据集并进行预处理

`数据预处理函数：`

```python
def preprocess(x, y):
    """
    预处理函数
    """
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.expand_dims(x, axis=3)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y
```

`加载手写数据集：`

```python
# 加载手写数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()

print(x.shape, y.shape, x_test.shape, y_test.shape)
```

`输出：`

```
(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
```

`转化为Dataset数据集：`

```python
batchsz = 1000


# 转化为Dataset数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y))

# 随机打散
train_db = train_db.shuffle(10000)

train_db = train_db.batch(batchsz)

# 数据预处理
train_db = train_db.map(preprocess)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.batch(batchsz).map(preprocess)

sample = sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)
print('batch:', sample[0].shape, sample[1].shape)
```

`输出：`

```
batch: (1000, 28, 28, 1) (1000, 10)
batch: (1000, 28, 28, 1) (1000, 10)
```

### 3.3 创建网络

```python
model = Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1, activation="relu"), # Conv Layer 1
    layers.MaxPool2D(pool_size=2, strides=2), # Pooling Layer 2
    layers.Conv2D(16, kernel_size=5, strides=1, activation="relu"), # Conv Layer 3
    layers.MaxPool2D(pool_size=2, strides=2), # Pooling Layer 4
    layers.Flatten(), # flatten 层，方便全连接处理
    layers.Dense(120, activation="relu"), # Fully connected layer 1
    layers.Dense(84, activation="relu"), # Fully connected layer 2
    layers.Dense(10) # Fully connected layer 
])
```

`打印网络结构：`

```python
model.build(input_shape=(None, 28, 28, 1))
model.summary()
```

`输出：`

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              multiple                  156       
_________________________________________________________________
max_pooling2d (MaxPooling2D) multiple                  0         
_________________________________________________________________
conv2d_1 (Conv2D)            multiple                  2416      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 multiple                  0         
_________________________________________________________________
flatten (Flatten)            multiple                  0         
_________________________________________________________________
dense (Dense)                multiple                  30840     
_________________________________________________________________
dense_1 (Dense)              multiple                  10164     
_________________________________________________________________
dense_2 (Dense)              multiple                  850       
=================================================================
Total params: 44,426
Trainable params: 44,426
Non-trainable params: 0
_________________________________________________________________
```

### 3.4 模型训练与验证

`模型装配：`

```python
model.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

`模型训练：`

```python
model.fit(
    train_db,
    epochs=5,
    validation_data=test_db,
    validation_freq=2
)
```

`输出：`

```
Train for 60 steps, validate for 10 steps
Epoch 1/5
60/60 [==============================] - 15s 253ms/step - loss: 1.0019 - accuracy: 0.7417
Epoch 2/5
60/60 [==============================] - 15s 246ms/step - loss: 0.2996 - accuracy: 0.9113 - val_loss: 0.2260 - val_accuracy: 0.9320
Epoch 3/5
60/60 [==============================] - 14s 232ms/step - loss: 0.2050 - accuracy: 0.9399
Epoch 4/5
60/60 [==============================] - 15s 243ms/step - loss: 0.1517 - accuracy: 0.9545 - val_loss: 0.1196 - val_accuracy: 0.9625
Epoch 5/5
60/60 [==============================] - 14s 228ms/step - loss: 0.1231 - accuracy: 0.9631
```

`模型验证：`

```python
model.evaluate(test_db)
```

`输出：`

```
10/10 [==============================] - 1s 71ms/step - loss: 0.0971 - accuracy: 0.9701
```

`转载自：` https://engmrk.com/lenet-5-a-classic-cnn-architecture/

![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/公众号宣传码.png)

