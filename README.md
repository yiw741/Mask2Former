# Mask2former

## 理论

![img](https://pica.zhimg.com/v2-6e9b00f7338ea3890f2b9607cd414b0c_r.jpg)

**掩码中的值是什么：多类掩码：类别标签**：在某些情况下，掩码的值可以是多个类别的标签。例如，掩码中的每个像素可以取不同的整数值，表示不同的类别（如 0 表示背景，1 表示人，2 表示车，3 表示树等）。这种形式常用于语义分割任务。

**Mask2Former的整体架构使用Transformer模型来提取图像特征，并使用反卷积网络将分辨率逐渐恢复到原始图像的大小。然后，使用Transformer解码器对图像特征进行操作以处理对象查询，并从每个位置的嵌入中解码二元掩码预测。**

Mask2Former 的架构包含了三个主要的模块：**遮盖编码器、遮盖解码器和蒸馏模块**。其中，遮盖编码器主要负责将输入图像编码为特征向量，**遮盖解码器则使用这些特征向量来生成分割掩模**，而蒸馏模块则用于进一步优化分割结果。



maskformer将语义分割看出是类别预测与mask预测两个子任务，类别预测和mask预测只要一一对应起来就可以解决语义分割任务.因此该任务的损失函数由两个组成，一个是预测分类图与真值图的损失，另一个是预测类别的交叉熵损失

![img](https://i-blog.csdnimg.cn/blog_migrate/45a005219932ccd89c7bfb93b001d929.png)

左边的图中表示了 基于每个位置用相同的分类损失的像素分类的语义分割

右边的图中表示基于掩码分类预测一组二值掩码，并为每个掩码分配一个类



首图中的01,02,03,04中输入的是由backone中的每一层的输出，而不是backbone最后的输出，如果仔细观察可以发现其实这四层之间并没有相连

- 在ResNet（Residual Network）架构中，网络通常被分为多个阶段（或称为层组），每个阶段由若干个残差块（Residual Block）组成。以ResNet50为例，它通常被划分为四个主要阶段：Res2、Res3、Res4和Res5。每个阶段的名称（如Res2）通常用于指代这一阶段中的所有残差块。

![image-20241125173458089](D:/Typora/image/image-20241125173458089.png)



------

Transformer module：**这里输入transformer中的就相当于q也就是条件，这里相当于cross attention**,q在一开始时没有进行初始化，相当于是0向量，所以一开始就给他们一个位置编码（按顺序），这就是他们的天赋，这时就去看他们哪个能够更好的对某一个分类的关注值更高，此就会使这个位置去专门去做这个任务的分类（这是在学习过程中一步步确定的），由于一开始会有超过分类类别的q所以多的就会是背景

在整个transformer经历完后，每个编码经过全连接层看看对那一类的分类更好,消去那些是空背景的，每个mask向量中的值代表该mask对图像中每个像素的预测概率或置信度，表示该像素属于该mask所代表的对象实例或类别的程度,**注意这里的mask已经是一个专门的二分类的值了，其中的值已经过softmax**,在下图红线中是一层全连接层，然后经过softmax

确定mask的分类目标过程

- **监督匹配**：
  - 在训练过程中，模型并不直接知道哪个查询对应哪个类别。相反，它使用一种匹配策略（如Hungarian算法）来将生成的掩码与真实的掩码进行最佳匹配。
- **监督信号**：
  - 使用真实的标签和掩码作为监督信号，指导模型学习如何生成与输入图像一致的掩码。
- **动态分配**：
  - 查询通过与特征交互和损失优化，逐渐学习到与特定类别相关的特征。这种关联是动态和数据驱动的。

![image-20241125230812345](D:/Typora/image/image-20241125230812345.png)

在segmentation module中每个工人进行预测，由于这里的mask代表是每一个像素点的概率，所直接对原图进行进行相乘，**内涵就是调整原图中的值是之目标分类值更大，之后输出的即是对应mask的分类图**，之后进行**损失计算（每个像素上是对应的值）**，所以这里有几个类就会有几个mask分别对原图进行相乘也就是调整，计算出对应的类的每个像素点的分类情况。在进行推理时，会以阈值（0.5）的形式将对于每个mask概率图进行二进制转换

![image-20241125231445711](D:/Typora/image/image-20241125231445711.png)

上方的是分类预测即当前mask所分的类是否正确，一个二分类是或不是，下方的则是比较精细的位置类，类似于iou分类

![image-20241125231730738](D:/Typora/image/image-20241125231730738.png)

------



**mask attention对于cross attention：**可以发现，在1中的q会对每一个像素点进行注意力计算，那么就会占用很多资源，所以我们此时就遮住某一些背景概率很大的点，以减少计算

所以我们现在在从backbone中逐层级传进来的深层到浅级特征，最小的浅级特征就去进行前景与背景的分割如res2，其他的就去进行transformer

有可能在每一次完成一个tansformer都会进行一次损失计算



这里的**decoder与DETR**中的差别一是mask attention，二是这里没有掩码对于输入的q进行遮掩，解码器输出用于生成掩码和类别标签。。借鉴于detr但解码器输出用于预测边界框和类别标签。

- **DETR**：
  - 解码器输出用于预测边界框和类别标签。
  - 输出的是一组对象查询（object queries），每个查询对应一个检测结果。
- **Mask2Former**：
  - 解码器输出用于**生成掩码和类别标签。**
  - 输出的是一组掩码查询（mask queries），**每个查询对应一个掩码。**
  - 每个查询生成一个**掩码概率图**（mask probability map），用于表示**图像中某个区域的分割结果**。
  - 使用掩码损失（如Dice loss或Focal loss）来优化掩码的质量。**损失函数也会进行判断哪个mask适合哪一个对象，每一个mask属于竞争关系**
  - 可能需要后处理步骤来合并或调整掩码。*Deformable Attention**



**Deformable Attention**

**Deformable Attention**可变形注意力的道理用大白话来说很简单：query不是和全局每个位置的key都计算注意力权重，而是**对于每个query，仅在全局位置中采样部分位置的key，并且value也是基于这些位置进行采样插值得到的**，最后将这个**局部&稀疏**的注意力权重施加在对应的value上。

可变形注意模块只关注参考点周围的一小部分关键采样点，而与特征图的空间大小无关，如图2所示，通过为每个查询仅分配少量固定数量的键，可以减轻收敛和[特征空间](https://ml-summit.org/cloud-member?uid=c1041&spm=1001.2101.3001.7020)分辨率的问题。

**以下过程会在每一个特征层都会进行一遍，输入是多个特征层**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/92ae492f23a78b04ca015cc8e743d086.png)

**偏置**

- 对于**每个注意力头（例如，Head 1、Head 2、Head 3）**，计算采样偏移 ΔpmqkΔ*p**m**q**k*。这些偏移量决定从输入特征图的哪些位置采样信息。
- 采样位置通过参考点和偏移量相加得到，允许注意力机制灵活地选择关注区域。

**注意力权重**

- 通过线性变换和softmax操作计算注意力权重 Amqk。这些权重衡量每个采样位置的相对重要性
- 权重的计算使得模型能够在不同位置之间分配不同的注意力程度。

**多头机制**：

- 每个头独立地执行采样和权重计算，提供多样化的特征表示。
- 这种多头机制允许模型在不同的空间位置和特征模式之间捕捉更多的细节。

**聚合**：

- 对每个头的采样值进行加权求和，得到聚合的采样值。
- 聚合操作整合了每个头的输出，为最终的输出提供丰富的特征信息。

**输出**：

- 所有头的聚合采样值经过线性变换，生成最终输出。
- 该输出结合了多头注意力机制的结果，具有更强的表达能力和空间适应性。

![img](https://pic4.zhimg.com/v2-d462c8b6ee4916db49bb84dee22d62db_1440w.jpg)

生成掩码特征后，通常会经过解码过程来产生最终的掩码输出，这一步骤可能包括：

- encoder生成掩码，decoder进行最终掩码输出

- 使用软最大（Softmax）函数来为每个像素分配一个类别标签。
- 后处理步骤，比如大津法（Otsu's method）或条件随机场（CRF），来优化掩码的边界。





















## 代码

### **MaskFormer类**：

- 这是整个模型的入口类，负责初始化模型的各个组件，如backbone、sem_seg_head和criterion。

![image-20241126231148389](D:/Typora/image/image-20241126231148389.png)















































### **MSDeformAttnPixelDecoder类**：



![image-20241127131537948](D:/Typora/image/image-20241127131537948.png)

#### MSDeformAttnPixelDecoder

```
transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
```

- input_shape.items()是所有可用的层，从backbone传来，transformer_in_features是所默认的层
- `{k: v for k,}`代表是输出v，在for前面的代表输出的数，在for后面的一般都是k,然后后面则是对前面要输出的进行限制，一般都是进行v的限制

假设：

```python
input_shape = {
    "res2": {"channel": 64, "stride": 8},
    "res3": {"channel": 128, "stride": 16},
    "res4": {"channel": 256, "stride": 32},
    "res5": {"channel": 512, "stride": 64},
}
transformer_in_features = ["res3", "res4", "res5"]
```

运行这行代码后，`transformer_input_shape` 将会是：

```python
{
    "res3": {"channel": 128, "stride": 16},
    "res4": {"channel": 256, "stride": 32},
    "res5": {"channel": 512, "stride": 64},
}
```

这样做可以确保后续模型只使用 "res3", "res4", "res5" 这些层而忽略 "res2"。







#### MSDeformAttnTransformerEncoderOnly

1. `MSDeformAttnTransformerEncoderLayer`：这是 Transformer 编码器的一层，包含自注意力模块 `self_attn` 和前馈网络模块 `ffn`。在进行自注意力运算时，使用了 MSDeformAttn 来实现。MSDeformAttn 是一个多尺度变形注意力模块，根据参考点计算注意力权重矩阵。
2. `MSDeformAttnTransformerEncoder`：这是 Transformer 编码器模块，包含多个 MSDeformAttnTransformerEncoderLayer 模块。
3. `MSDeformAttnTransformerEncoderOnly`：这是一个只包含 Transformer 编码器的模型，继承自 `nn.Module`。在构造函数中，首先创建了一个自注意力模块 `transformer`，然后处理输入特征，使它们可以输入至自注意力模块进行计算。
4. `MSDeformAttnPixelDecoder`：这是一个由 Transformer 编码器和 FPN 网络构成的模型，主要用于对象检测任务。通过在对应特征层计算注意力权重，模型学习到了各个特征层上的重要信息。

##### MSDeformAttnTransformerEncoderLayer

**MSDeformAttn**

`MSDeformAttn` 类实现了一个多尺度变形注意力模块。这个模块在处理图像和序列数据时，会对不同尺度的特征进行动态采样，从而提升模型的灵活性和适应性。它通过动态计算采样位置和结合注意力权重来加强特征提取过程。

ms_deform_attn_func

多尺度变形注意力机制的核心计算函数，实现跨尺度、跨位置的自适应特征采样和加权聚合

























### **MultiScaleMaskedTransformerDecoder类**：

![image-20241127125035830](D:/Typora/image/image-20241127125035830.png)

每一层都独立处理自己的特征图，输出结果经过一系列的注意力机制和前馈神经网络，最终每一层的输出被存储并可以用于后续的预测。模型设计如此，允许模型在不同层中逐渐融合信息，从而提高检测精度和特征提取能力。

类别嵌入（Category Embedding）是深度学习中一种常用的表示技巧，用于将离散类别数据（如分类标签、类别名称等）转换为连续的低维向量表示。这种嵌入可以使模型更好地处理类别信息，并且能在许多任务中提高性能，尤其是在推荐系统、自然语言处理和计算机视觉等领域。









































### **其他关键类**：

- 如`Matcher`类，用于计算损失时的匹配算法。
- `Losses`类，用于计算分类损失和掩码损失。



































