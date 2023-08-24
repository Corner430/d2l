1. [CPU 和 GPU](1.CPU和GPU.ipynb)
    - 如何提升 GPU 和 CPU 的利用率
    - 不要频繁在 CPU 和 GPU 之间传输数据
    - 80% 的论文无法复现
    - resnet不可以用在文本上
2. [TPU 和 其他](2.TPU和其他.ipynb)
    - Google TPU
    - Systolic Array
    - 功耗不是问题，电厂也不缺电
    - 做芯片的风险
    - GPU 和 网络要互相 match
    - **Transformer 非常适合 TPU，都是 Google 家的**
3. [单机多卡并行](3.单机多卡并行.ipynb)
    - 数据并行和模型并行
4. [多 GPU 训练实现](4.多GPU训练实现.ipynb)
    - 数据并行时，每个GPU会得到所有的参数
5. [分布式训练](5.分布式训练.ipynb)
    - batch_size 通常不要大于 10 x class_num