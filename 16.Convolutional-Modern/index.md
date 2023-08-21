1. [AlexNet](1.AlexNet.ipynb)
    - AlexNet的出现
2. [代码](2.代码.ipynb)
3. [QA](3.QA.ipynb)
4. [VGG](4.VGG.ipynb)
    - 更大更深的AlexNet（重复的VGG块）
    - 进度
5. [代码](5.代码.ipynb)
    - 深层且窄的卷积（即$3 \times 3$）比较浅层且宽的卷积更有效
    - 与AlexNet相比，**VGG的计算要慢得多**，而且它还需要更多的显存
6. [QA](6.QA.ipynb)
7. [NiN](7.NiN.ipynb)
    - **卷积层后的第一个全连接层带来的问题**
    -  **NiN架构**
        - 无全连接层
        - 最后使用**全局平均池化层**得到输出，**其输入通道数是类别数**，不容易过拟合，更少的参数个数
8. [代码](8.代码.ipynb)
9. [QA](9.QA.ipynb)