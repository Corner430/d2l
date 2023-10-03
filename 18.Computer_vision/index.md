1. [image augmentation](1.image_augmentation.ipynb)
    - 图像增强一可以扩大数据集，二可以提高模型的泛化能力
    - 可以认为是一种正则
    - [imgaug](https://github.com/aleju/imgaug)
    - cifar-10跑到95、98也没什么问题，一般要 200 个 epoch，但是 gap 很难随着 epoch 减少了
    - **图像增强之后，可以输出看一眼**
    - **图像增强之后一般不会改变分布**
    - mix-up增广
2. [fine-tuning](2.fine_tuning.ipynb)
    - 训练是一个目标数据集上的正常训练任务，**但使用更强的正则化**
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
3. [Kaggle leaf](3.Kaggle_leaf.ipynb)
4. [Kaggle Cifar-10](4.Kaggle_Cifar_10.ipynb)
    - 不同的类别，放在不同的文件夹中
    - `torch.optim.lr_scheduler.StepLR()`，每隔一定的 epoch，学习率乘以一个系数
    - `drop_last=True`，如果最后一个 batch 的样本数不足 batch_size，就丢弃
    - `momentum`
    - 常见的 `scheduler`
5. [Kaggle ImageNet dog classification](5.Kaggle_ImageNet_dog.ipynb)
    - `scale` 和 `ratio` 的作用是什么？
6. [Bound box](6.bound_box.ipynb)
7. [Anchor](7.anchor.ipynb)
    - 提出多个被称为锚框的区域（边缘框），**预测每个锚框里是否含有关注的物体，如果是，预测从这个锚框到真实边缘框的偏移**
    - 非极大值抑制（NMS）
    - 做 NMS 时有两种，一种是对所有类，一种是针对每一个类别
    - 锚框生成的细节
    - 交并比（IoU）
    - **在训练集中，我们需要给每个锚框两种类型的标签**，一种是与锚框中目标检测的类别（class），另一种是锚框相对于真实边界框的偏移量（offset）。
    - **将真实边界框分配给锚框**
8. [Object Detection Dataset](8.object-detection-dataset.ipynb)
    - 目标检测的 label 不太一样
    - 小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中。
    - `permute` 维度重排
9. [Multiscale object detection](9.multiscale-object-detection.ipynb)
    - **当使用较小的锚框检测较小的物体时，我们可以采样更多的区域，而对于较大的物体，我们可以采样较少的区域。**
    - **简言之，我们可以利用深层神经网络在多个层次上对图像进行分层表示，从而实现多尺度目标检测。**
10. [SSD](10.ssd.ipynb)
    - **SSD 在多个段的输出上进行多尺度的检测**
    - L1 范数损失
    - **平滑 L1 范数损失：当$\sigma$非常大时，这种损失类似于$L_1$范数损失。当它的值较小时，损失函数较平滑。**
    - **焦点损失：增大$\gamma$可以有效地减少正类预测概率较大时（例如$p_j > 0.5$）的相对损失，因此训练可以更集中在那些错误分类的困难示例上。**
11. [RCNN](11.rcnn.ipynb)
    - RCNN, Fast RCNN, Faster RCNN, Mask RCNN
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
12. [YOLO](12.YOLO.ipynb)
13. [Semantic Segmentation and Dataset](13.semantic-segmentation-and-dataset.ipynb)
    - 语义分割、图像分割、实例分割
    - Pascal VOC2012数据集
    - **由于语义分割的输入图像和标签在像素上一一对应，输入图像会被随机裁剪为固定尺寸而不是缩放。**
    - 语义分割标注工具
    - 三维的语义分割
14. [Transposed Conv](14.transposed-conv.ipynb)
    - **转置卷积其实是一种上采样技术**
    - **恢复卷积前的图像尺寸，而不是恢复原始值。**
    - **为什么称之为“转置”**
    - 高级 API `nn.ConvTranspose2d`
    - 转置卷积中的 步幅 和 填充
    - **可以用矩阵乘法来实现卷积和转置卷积**
    - **[转置卷积是一种卷积](https://www.bilibili.com/video/BV1CM4y1K7r7/?spm_id_from=autoNext&vd_source=2dd00fcea46a9c5a26706a99eb12ea3f)**
15. [Fcn](15.fcn.ipynb)
16. [Neural Style](16.neural-style.ipynb)