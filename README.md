# d2l 笔记

> **由于Github数据库抽风的问题，如果点链接出现了`not found`。请先点开`readme`，再点链接**

> 课程学习需要资料汇总

- [动手学深度学习（Dive into Deep Learning，D2L.ai）Github](https://github.com/d2l-ai/d2l-zh)
- [D2L电子书](https://zh.d2l.ai/)
- [D2L电子书英文版](https://d2l.ai/)：英文版含有更多的内容
- [D2L电子书英文版PDF](https://d2l.ai/d2l-en.pdf)
- [基础数学知识](http://www.d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)
- [讨论区](https://discuss.d2l.ai/)
- [讨论区中文版](https://discuss.d2l.ai/c/16)
- [Distill](https://distill.pub/)
- [Python教程](http://learnpython.org/)<!--more-->

----------------------------------
> 《动手学深度学习（PyTorch版）》配套资源获取链接：
- [本书配套网站主页](https://d2l.ai/)
- [课程主页](https://courses.d2l.ai/zh-v2)
- [教材](https://zh-v2.d2l.ai/)
- [Pytroch论坛](https://discuss.pytorch.org/)
- [GitHub项目地址](https://github.com/d2l-ai/d2l-zh)
- [Jupyter记事本下载](https://zh-v2.d2l.ai/d2l-zh.zip)
- [中文版课件](https://github.com/d2l-ai/berkeley-stat-157/tree/master/slides-zh)
- [视频课程及课程PPT](https://courses.d2l.ai/zh-v2/)
- 习题：见纸书
- 社区讨论：见纸书各节二维码

-------------------------------------
## 搭配读物
- [pytorch-handbook](https://github.com/zergtant/pytorch-handbook)
- [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
- [machine_learning_beginner 中的 python 基础](https://github.com/fengdu78/machine_learning_beginner)
  - 1.[两天入门Python(目录名：python-start)](https://github.com/fengdu78/machine_learning_beginner/blob/master/python-start)
  - 2.[适合初学者快速入门的Numpy实战全集(目录名：numpy)](https://github.com/fengdu78/machine_learning_beginner/blob/master/numpy)
  - 3.[matplotlib学习之基本使用(目录名：matplotlib)](https://github.com/fengdu78/machine_learning_beginner/blob/master/matplotlib)
  - 4.[两天学会pandas(目录名：pandas)](https://github.com/fengdu78/machine_learning_beginner/blob/master/pandas)

----------------------------------------
[GluonCV Model Zoo](https://cv.gluon.ai/model_zoo/classification.html)

## 目录
1. [basic knowledge](1.basic_knowledge/index.md)
    - [环境搭建](1.basic_knowledge/1.环境搭建.ipynb)
    - [数据操作](1.basic_knowledge/2.数据操作.ipynb)
        - 基本的数据操作以及csv文件的读取
        - 数据的处理
    - [线性代数](1.basic_knowledge/3.线性代数.ipynb)
        - 矩阵的基本操作
        - 矩阵的基本知识
    - [微积分](1.basic_knowledge/4.微积分.ipynb)
        - 四种求导
        - 画图函数
    - [自动求导](1.basic_knowledge/5.自动求导.ipynb)
        - 对微积分知识的应用
    - [概率论](1.basic_knowledge/6.概率论.ipynb)
        - 概率的反直觉
    - [查阅文档](1.basic_knowledge/7.查阅文档.ipynb)
        - 查找模块中所有的函数和类
        - 查找特定函数和类的用法
2. [Linear regression](2.Linear_regression/index.md)
    - **如何定义一个`data_iter`，用于生成`batch_size`大小的数据**
    - **如何使用`torch.utils.data`中的`TensorDataset`和`DataLoader`来读取数据**
    - 选择合适的`batch_size`
    - **样本大小不是批量大小的整数倍怎么办？**
3. [Softmax regression](3.Softmax_regression/index.md)
    - Softmax回归理论基础
    - 图像分类数据集（Fashion-MNIST）
    - 解决softmax中的数值上下溢的问题
4. [Multilayer perceptron](4.multilayer_perceptron/index.md)
    - 感知机理论基础
      - 二分类
      - 收敛定理
    - 多层感知机理论知识
      - 解决异或问题
      - 各种激活函数
      - 通用近似定理
    - 多层感知机的从零开始实现
      - `nn.Parameter()`、`@`
    - 多层感知机的简洁实现]
    - QA
      - 为什么玩深度学习，而不是广度学习？
      - 相对而言，激活函数并不是很重要
      - 选择2的幂次作为batch_size的原因
5. [Model selection Overfitting and underfitting](5.Model_selection_overfitting_and_underfitting/index.md)
    - K折交叉验证
    - VC维
    - 多项式拟合
    - 训练集、验证集和测试集
6. [Weight decay](6.Weight_decay/index.md)
    - 权重衰退所能带来的效果很有限
    - $L_1$范数和$L_2$范数的区别
7. [dropout](7.dropout/index.md)
    - **当面对更多的特征而样本不足时，线性模型往往会过拟合**
    - **无偏**的加入噪声
    - 一般取值为**0.1、0.5、0.9**
    - **可以将模型设置的复杂一些，之后使用dropout**
    - n个数字相加，加的顺序不一样，结果会不一样
    - dropout和regularization可以**同时使用**
    - 随机种子
    - **dropout可能会导致参数收敛变慢**
8. [backprop](8.backprop.ipynb/index.md)
    - 前向传播、方向传播和计算图
    - 分布式训练
9. [numerical stability and init](9.numerical_stability_and_init/index.md)
    - 梯度消失和梯度爆炸
    - 对称性问题
    - 如何让训练更加稳定
    - Xavier初始化
    - 从泰勒的角度去理解激活函数
    - **合理的权重初始值和激活函数的选取可以提升数值稳定性**
    - `nan`、`inf`
10. [enviroment and distribution shift](10.environment_and_distribution_shift/index.md)
    - 环境和分布偏移以及纠正方式
11. [Kaggle实战之房价预测](11.Kaggle_predict_house_price/index.md)
    - **[autogluon](https://auto.gluon.ai/stable/index.html)也可以取得很好的效果**
12. [deep learning computation](12.deep_learning_computation/index.md)
    - 模型构造
    - 参数管理
    - 自定义层
    - 读写文件
13. [GPU](13.GPU/index.md)
14. [CNN](14.CNN/index.md)
    - **从全连接层到卷积层**
    - 交叉相关和卷积
    - **卷积层就是一个特殊的全连接层**
    - 训练抖动
    - **填充**在输入周围添加额外的行/列，**来控制输出形状的减少量**
    - **步幅**是在每次滑动核窗口时的行/列的步长，**可以成倍的减少输出形状**
    - kernel_size、padding 和 strides 怎么设置
    - kernel_size一般选奇数
    - **NAS 可以让超参数也一起训练**
    - **机器学习本质上就是在做压缩**
    - 为什么要有多个输入输出通道？
    - **1 x 1卷积层的作用**
    - **输出通道数**是卷积层的超参数
    - **每个输入通道有独立的二维卷积核，所有通道结果相加得到一个输出通道结果**
    - **每个输出通道有独立的三维卷积核**
    - **通道数如何确定**
    - padding 0 对输出的影响
    - 卷积->池化
    - 池化层为什么用的越来越少了？
    - 感受野
15. [Classical convolution neural network LeNet](15.LeNet/index.md)
    - `__class__.__name__`
    - 卷积可视化：[CNN Explainer](https://poloclub.github.io/cnn-explainer/)
16. [Convolutional Modern](16.Convolutional-Modern/index.md)
    - AlexNet
    - VGG：**更大更深的AlexNet（重复的VGG块）**
    - 深层且窄的卷积（即$3 \times 3$）比较浅层且宽的卷积更有效
    - 与AlexNet相比，**VGG的计算要慢得多**，而且它还需要更多的显存
    - **卷积层后的第一个全连接层带来的问题**
    -  **NiN架构**
        - 无全连接层
        - 最后使用**全局平均池化层**得到输出，**其输入通道数是类别数**，不容易过拟合，更少的参数个数
    - NiN 收敛变慢
    - **NiN 在每个像素的通道上分别使用多层感知机**
    - GoogLeNet 将各种卷积超参数都用上了，是一个**含并行连结的网络**
    - GoogLeNet 的一个**主要优点是模型参数小，计算复杂度低**
    - 为什么 GoogLeNet 这个网络如此有效呢？
    - Inception 块相当于一个有 4 条路径的子网络
    - Inception 块的通道数分配之比是在 ImageNet 数据集上通过大量的实验得来的
    - **批量归一化层解决的问题**
    - **批量归一化层作用范围**
    - **批量归一化层在卷积层为什么作用在通道维度**
    - **批量归一化在做什么**
    - **可以加速收敛速度，但一般不改变模型精度**
    - 方差的计算
    - **指数加权平均**
    - **Batch Normalization 对 `batch_size` 具有一定敏感性**
    - **标准化（Normalization）、归一化（Scaling）和 批量归一化（Batch Normalization）的区别**
    - **批量归一化中 $\gamma$ 和 $\beta$ 的作用**
    - **ResNet 解决了模型偏差的问题，使用了模型（函数）嵌套。也就是扩大了函数类。**
    - **学习率调度**
    - 所谓残差
    - ResNet为什么能训练出1000层的模型？
    - DenseNet
    - 稠密块的**核心思想是：每个卷积层的输入都包含了前面所有层的输出**。这意味着，每个卷积层都能够访问到网络中所有层的信息，从而增强特征的重用和梯度流动。
17. [Computation performance](17.Computation-performance/index.md)
    - 如何提升 GPU 和 CPU 的利用率
    - 不要频繁在 CPU 和 GPU 之间传输数据
    - **80% 的论文无法复现**
    - resnet不可以用在文本上
    - Google TPU
    - Systolic Array
    - 功耗不是问题，电厂也不缺电
    - 做芯片的风险
    - GPU 和 网络要互相 match
    - **Transformer 非常适合 TPU，都是 Google 家的**
    - 数据并行和模型并行
    - 数据并行时，每个GPU会得到所有的参数
    - 小批量数据量更大时，学习率也需要稍微提高一些。
    - **计算所需的时间长于同步参数所需的时间，并行化开销的相关性较小时，并行化才能体现出优势**
    - batch_size 通常不要大于 10 x class_num
    - 命令式编程和符号式编程
    - 保存模型到磁盘
    - 异步计算
    - 前后端
    - 通常情况下单个操作符将使用所有CPU或单个GPU上的所有计算资源
    - 值得注意的是，在实践中经常会有这样一个判别：加速卡是为训练还是推断而优化的
    - 设备有运行开销。因此，**数据传输要争取量大次少而不是量少次多。这适用于RAM、固态驱动器、网络和GPU**。
    - **矢量化是性能的关键**
    - **在训练过程中数据类型过小导致的数值溢出可能是个问题（在推断过程中则影响不大）**。
    - **训练硬件和推断硬件在性能和价格方面有不同的优点**。
18. [Computer vision](18.Computer_vision/index.md)
    - 图像增强一可以扩大数据集，二可以提高模型的泛化能力
    - 可以认为是一种正则
    - [imgaug](https://github.com/aleju/imgaug)
    - cifar-10跑到95、98也没什么问题，一般要 200 个 epoch，但是 gap 很难随着 epoch 减少了
    - **图像增强之后，可以输出看一眼**
    - **图像增强之后一般不会改变分布**
    - mix-up增广
    - 微调训练是一个目标数据集上的正常训练任务，**但使用更强的正则化**
        - 使用**更小的学习率**
        - 使用更少的数据迭代
    - 源数据集远复杂于目标数据，通常**微调效果更好**
    - **源数据集可能也有目标数据中的部分标号，可以使用预训练好模型分类器中对应标号对应的向量来做初始化**
    - **可以固定底部一些层的参数，不参与更新，这相当于一种更强的正则**
    - **预训练模型质量很重要**，并且微调通常速度更快，**精度更高**
    - **对于不同的层，使用不同的学习率**
    - 找合适的 pre-trained model
    - **微调中的归一化很重要，但是如果说模型中有 BatchNorm 的话，其实是相同作用的。也就是说，可以将归一化看作是模型架构的一部分**
    - **微调通常不会让模型变差**
    - 迁移学习将从源数据集中学习的知识迁移到目标数据集，**微调是迁移学习的常见技巧**
    - 不同的类别，放在不同的文件夹中
    - `torch.optim.lr_scheduler.StepLR()`，每隔一定的 epoch，学习率乘以一个系数
    - `drop_last=True`，如果最后一个 batch 的样本数不足 batch_size，就丢弃
    - `momentum`
    - 常见的 `scheduler`
    - `scale` 和 `ratio` 的作用是什么？
    - 提出多个被称为锚框的区域（边缘框），**预测每个锚框里是否含有关注的物体，如果是，预测从这个锚框到真实边缘框的偏移**
    - 非极大值抑制（NMS）
    - 做 NMS 时有两种，一种是对所有类，一种是针对每一个类别
    - 锚框生成的细节
    - 交并比（IoU）
    - **在训练集中，我们需要给每个锚框两种类型的标签**，一种是与锚框中目标检测的类别（class），另一种是锚框相对于真实边界框的偏移量（offset）。
    - **将真实边界框分配给锚框**
    - 目标检测的 label 不太一样
    - 小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中。
    - `permute` 维度重排
    - **当使用较小的锚框检测较小的物体时，我们可以采样更多的区域，而对于较大的物体，我们可以采样较少的区域。**
    - **简言之，我们可以利用深层神经网络在多个层次上对图像进行分层表示，从而实现多尺度目标检测。**
    - RCNN, Fast RCNN, Faster RCNN, Mask RCNN
    - **SSD 在多个段的输出上进行多尺度的检测**
    - **SSD 在多个段的输出上进行多尺度的检测**
    - L1 范数损失
    - **平滑 L1 范数损失：当$\sigma$非常大时，这种损失类似于$L_1$范数损失。当它的值较小时，损失函数较平滑。**
    - **焦点损失：增大$\gamma$可以有效地减少正类预测概率较大时（例如$p_j > 0.5$）的相对损失，因此训练可以更集中在那些错误分类的困难示例上。**
    - RCNN
        - 使用启发式搜索算法来选择锚框
        - 使用预训练模型来对**每个锚框**抽取特征
        - 训练一个 SVM 来对类别分类
        - 训练一个线性回归模型来预测边缘框偏移
    - Fast RCNN
        - **R-CNN的主要性能瓶颈在于，对每个提议区域，卷积神经网络的前向传播是独立的，而没有共享计算。**
        - 由于这些区域**通常有重叠**，独立的特征抽取会导致重复的计算。
        - **Fast R-CNN 对R-CNN的主要改进之一，是仅在整张图象上执行卷积神经网络的前向传播。**
    - Faster RCNN
      - 使用一个区域提议网络来替代启发式搜索来获得更好的锚框
    - * Mask R-CNN在Faster R-CNN的基础上引入了一个全卷积网络，从而**借助目标的像素级位置进一步提升目标检测的精度。**
    - YOLO
    - 语义分割、图像分割、实例分割
    - Pascal VOC2012数据集
    - **由于语义分割的输入图像和标签在像素上一一对应，输入图像会被随机裁剪为固定尺寸而不是缩放。**
    - 语义分割标注工具
    - 三维的语义分割
    - **转置卷积其实是一种上采样技术**
    - **恢复卷积前的图像尺寸，而不是恢复原始值。**
    - **为什么称之为“转置”**
    - 高级 API `nn.ConvTranspose2d`
    - 转置卷积中的 步幅 和 填充
    - **可以用矩阵乘法来实现卷积和转置卷积**
    - **[转置卷积是一种卷积](https://www.bilibili.com/video/BV1CM4y1K7r7/?spm_id_from=autoNext&vd_source=2dd00fcea46a9c5a26706a99eb12ea3f)**
    - **FCN用转置卷积层来替换 CNN 最后的全连接层，从而可以实现每个像素的预测**
    - `list(pretrained_net.children())[-3:]`
    - **全卷积网络先使用卷积神经网络抽取图像特征，然后通过$1\times 1$卷积层将通道数变换为类别个数，最后通过转置卷积层将特征图的高和宽变换为输入图像的尺寸。**
    - 在全卷积网络中，我们**可以将转置卷积层初始化为双线性插值的上采样。**
    - `net.add_module`
    - **风格迁移常用的损失函数由3部分组成：**
      - **内容损失**使合成图像与内容图像在内容特征上接近；
      - **风格损失**令合成图像与风格图像在风格特征上接近；
      - **全变分损失**则有助于减少合成图像中的噪点。
    - 我们可以通过预训练的卷积神经网络来抽取图像的特征，并通过最小化损失函数来不断更新合成图像来作为模型参数。
    - 我们使用**格拉姆矩阵**表达风格层输出的风格。
19. [Recurrent Neural Networks](19.recurrent_neural_networks/index.md)
    - 时序模型中，当前数据跟之前观察到的数据相关
    - 自回归模型使用自身过去数据来预测未来
    - 马尔可夫模型假设当前只跟最近少数数据相关，从而简化模型
    - 潜变量模型使用潜变量来概括历史信息
    - 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进行训练。
    - 序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型。
    - 对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。
    - **对于直到时间步$t$的观测序列，其在时间步$t+k$的预测输出是“$k$步预测”。随着我们对预测时间$k$值的增加，会造成误差的快速累积和预测质量的极速下降。**
    - $tau$ 并不是越大越好，极端来说，当 $tau$ 等于序列长度时，就只有一个样本，这样就没有意义了。
    - **H.G.Well的[时光机器](https://www.gutenberg.org/ebooks/35)**
    - 文本预处理
    - 马尔可夫模型与 n 元语法
    - **读取长序列的主要方式是随机采样和顺序分区**
    - **隐藏层和隐状态指的是两个截然不同的概念。**
    - **循环神经网络模型的参数数量不会随着时间步的增加而增加。**
    - **我们可以使用困惑度来评价语言模型的质量。**
    - 更多的应用 RNNs
    - **循环神经网络模型在训练以前需要初始化状态，不过随机抽样和顺序划分使用初始化方法不同。**
    - **当使用顺序划分时，我们需要分离梯度以减少计算量。**
    - **在进行任何预测之前，模型通过预热期进行自我更新（例如，获得比初始值更好的隐状态）。**
    - **梯度裁剪可以防止梯度爆炸，但不能应对梯度消失。**
    - **高级API的循环神经网络层返回一个输出和一个更新后的隐状态，我们还需要计算整个模型的输出层。**
    - **相比从零开始实现的循环神经网络，使用高级API实现可以加速训练。**