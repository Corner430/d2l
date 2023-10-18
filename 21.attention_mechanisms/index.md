1. [Attention Cues](1.attention-cues.ipynb)
    - 注意力可视化，热图（heatmap）
2. [Nadaraya waston](2.nadaraya-waston.ipynb)
    - `torch.bmm()`函数实现了批量矩阵乘法
    - 注意力汇聚：Nadaraya-Watson 核回归
    - 平均汇聚
    - 非参数注意力汇聚
    - 带参数注意力汇聚
3. [Attention Scoring Functions](3.attention-scoring-functions.ipynb)
    - 注意力评分函数
    - 掩码 softmax 操作
    - 当查询和键是不同长度的矢量时，也可以使用**加性注意力**作为评分函数。
    - 缩放点积注意力
    - 点积操作要求查询和键具有相同的长度$d$。**假设查询和键的所有元素都是独立的随机变量，并且都满足零均值和单位方差，**那么两个向量的点积的均值为$0$，方差为$d$。**为确保无论向量长度如何，点积的方差在不考虑向量长度的情况下仍然是$1$，我们再将点积除以$\sqrt{d}$，**
4. [Bahdanau Attention](4.bahdanau-attention.ipynb)
    - 注意力编码器
5. [Multihead Attention](5.multihead-attention.ipynb)
    - **多头注意力融合了来自于多个注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的子空间表示。**
    - **基于适当的张量操作，可以实现多头注意力的并行计算。**
6. [Self Attention and Positional Encoding](6.self-attention-and-positional-encoding.ipynb)
    - 在自注意力中，查询、键和值都来自同一组输入。
    - 卷积神经网络和自注意力都拥有并行计算的优势，而且自注意力的最大路径长度最短。但是因为其计算复杂度是关于序列长度的二次方，所以在很长的序列中计算会非常慢。
    - **为了使用序列的顺序信息，可以通过在输入表示中添加位置编码，来注入绝对的或相对的位置信息。**
7. [Transformer](7.transformer.ipynb)
    - Transformer模型完全基于注意力机制，没有任何卷积层或循环神经网络层
    - 层规范化和批量规范化的目标相同，但层规范化是基于特征维度进行规范化。**层规范化更适用于自然语言处理**