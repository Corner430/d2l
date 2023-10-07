1. [Sequence](1.sequence.ipynb)
    - 时序模型中，当前数据跟之前观察到的数据相关
    - 自回归模型使用自身过去数据来预测未来
    - 马尔可夫模型假设当前只跟最近少数数据相关，从而简化模型
    - 潜变量模型使用潜变量来概括历史信息
    - 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进行训练。
    - 序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型。
    - 对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。
    - **对于直到时间步$t$的观测序列，其在时间步$t+k$的预测输出是“$k$步预测”。随着我们对预测时间$k$值的增加，会造成误差的快速累积和预测质量的极速下降。**
    - $tau$ 并不是越大越好，极端来说，当 $tau$ 等于序列长度时，就只有一个样本，这样就没有意义了。
2. [Text Preprocessing](2.text-preprocessing.ipynb)
    - **H.G.Well的[时光机器](https://www.gutenberg.org/ebooks/35)**
3. [Lauguage Model and Dataset](3.language-models-and-dataset.ipynb)
    - 马尔可夫模型与 n 元语法
    - 单词的频率满足**齐普夫定律（Zipf's law）**
    - **读取长序列的主要方式是随机采样和顺序分区**
4. [RNN](4.rnn.ipynb)
    - **隐藏层和隐状态指的是两个截然不同的概念。**
    - **循环神经网络模型的参数数量不会随着时间步的增加而增加。**
    - **我们可以使用困惑度来评价语言模型的质量。**
    - [梯度裁剪](https://www.bilibili.com/video/BV1D64y1z7CA/?share_source=copy_web&vd_source=a7ae9163cb2cd121bfd86ea1f4ecd2ef&t=929)
    - [更多的应用 RNNs](https://www.bilibili.com/video/BV1D64y1z7CA/?share_source=copy_web&vd_source=a7ae9163cb2cd121bfd86ea1f4ecd2ef&t=1142)
5. [Rnn Scratch](5.rnn-scratch.ipynb)
    - 预热
6. [Rnn Concise](6.rnn-concise.ipynb)