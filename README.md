# d2l 笔记

> 课程学习需要资料汇总

[动手学深度学习（Dive into Deep Learning，D2L.ai）Github](https://github.com/d2l-ai/d2l-zh)
[D2L电子书](https://zh.d2l.ai/)
[D2L电子书英文版](https://d2l.ai/)：英文版含有更多的内容
[D2L电子书英文版PDF](https://d2l.ai/d2l-en.pdf)
[基础数学知识](http://www.d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)
[讨论区](https://discuss.d2l.ai/)
[讨论区中文版](https://discuss.d2l.ai/c/16)
[Distill](https://distill.pub/)
[Python教程](http://learnpython.org/)<!--more-->

----------------------------------
《动手学深度学习（PyTorch版）》配套资源获取链接：
[本书配套网站主页](https://d2l.ai/)
[课程主页](https://courses.d2l.ai/zh-v2)
[教材](https://zh-v2.d2l.ai/)
[Pytroch论坛](https://discuss.pytorch.org/)
[GitHub项目地址](https://github.com/d2l-ai/d2l-zh)
[Jupyter记事本下载](https://zh-v2.d2l.ai/d2l-zh.zip)
[中文版课件](https://github.com/d2l-ai/berkeley-stat-157/tree/master/slides-zh)
[视频课程及课程PPT](https://courses.d2l.ai/zh-v2/)
习题：见纸书
社区讨论：见纸书各节二维码

-------------------------------------
#### 搭配读物
- [pytorch-handbook](https://github.com/zergtant/pytorch-handbook)
- [machine_learning_beginner 中的 python 基础](https://github.com/fengdu78/machine_learning_beginner)
  - 1.[两天入门Python(目录名：python-start)](https://github.com/fengdu78/machine_learning_beginner/blob/master/python-start)
  - 2.[适合初学者快速入门的Numpy实战全集(目录名：numpy)](https://github.com/fengdu78/machine_learning_beginner/blob/master/numpy)
  - 3.[matplotlib学习之基本使用(目录名：matplotlib)](https://github.com/fengdu78/machine_learning_beginner/blob/master/matplotlib)
  - 4.[两天学会pandas(目录名：pandas)](https://github.com/fengdu78/machine_learning_beginner/blob/master/pandas)

----------------------------------------

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
    - [查阅文档](1.basic_knowledge/6.查阅文档.ipynb)
        - 查找模块中所有的函数和类
        - 查找特定函数和类的用法
2. [Linear regression](2.Linear_regression/index.md)
    - **如何定义一个`data_iter`，用于生成`batch_size`大小的数据**
    - **如何使用`torch.utils.data`中的`TensorDataset`和`DataLoader`来读取数据**
3. [Softmax regression](3.Softmax_regression/index.md)
    - Softmax回归理论基础
    - 图像分类数据集（Fashion-MNIST）
    - 解决softmax中的数值上下溢的问题
