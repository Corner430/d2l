1. [image augmentation](1.image_augmentation.ipynb)
    - 图像增强一可以扩大数据集，二可以提高模型的泛化能力
    - 可以认为是一种正则
    - [imgaug](https://github.com/aleju/imgaug)
    - cifar-10跑到95、98也没什么问题，一般要 200 个 epoch，但是 gap 很难随着 epoch 减少了
    - **图像增强之后，可以输出看一眼**
    - **图像增强之后一般不会改变分布**
    - mix-up增广
2. [fine-tuning](2.fine_tuning.ipynb)