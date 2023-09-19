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
    - 锚框生成的细节
    - 交并比（IoU）
    - 每个锚框的类别（class）和偏移量（offset）
    - **将真实边界框分配给锚框**
8. [Object Detection Dataset](8.object-detection-dataset.ipynb)
    - 目标检测的 label 不太一样
    - 小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中。