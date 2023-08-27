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