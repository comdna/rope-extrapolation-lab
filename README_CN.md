# 怎样调整位置编码，使得大模型具有更好的外推能力？

[English](README.md) | **中文** | [实验与复现](EXPERIMENTS_CN.md)

## Why Valuable

很多开源模型的技术报告有PE和外推能力的部分

大幅降低长上下文模型的训练成本，实现"训短推长"

解锁长文本核心应用场景，直接提升模型实用能力

深化对Transformer位置信息编码机制的理论认知，推动架构演进

## Intro

外推能力：大模型在训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了512个 token 的文本，那么在预测时如果输入超过512个 token，模型可能无法正确处理。这就限制了大模型在处理长文本或多轮对话等任务时的效果。

为什么位置编码会影响llm的外推能力：Transformer架构本身不具备感知token顺序的能力，必须依赖位置编码来注入序列中的位置信息，这使得模型对位置信号产生了根本性的依赖；而模型在预训练时只见过有限长度内的位置编码模式，其注意力权重和特征表示都是针对这一范围内的位置分布学到的，本质上是在一个"有界区间"内拟合。因此，当推理时序列长度超出训练范围，位置编码会生成模型从未见过的数值或向量，陌生的位置信号落入训练分布之外，导致注意力分数异常、输出质量急剧退化。位置编码方案对位置信息的编码方式决定了超出训练长度后信号的"可预测性"，故会影响外推能力。

## 应用现状和问题

RoPE对Transformer 每一层采用相同的 RoPE 缩放规则，没有区分浅层与深层。这种统一处理无法同时满足浅层“精确定位”和深层“弱化位置约束”的不同需求，当上下文长度远超预训练样本的长度时，难以兼顾局部结构建模与全局语义整合，最终导致长上下文检索或推理性能下降。

统一缩放会形成两种相互冲突的因果链：
1.缩放较强时：对所有层统一压缩 RoPE，造成浅层细粒度相对位置信息变弱；再造成浅层无法正确组合邻近词元、构建局部结构并将长序列锚定到预训练语义空间；最后造成困惑度急剧上升、长文本检索性能崩溃。
2.缩放较弱时：为了保护浅层而在所有层保留较强位置信号，造成深层仍受到过于僵化的位置约束；再造成位置噪声干扰深层对语义表征的全局整合与组合推理；最后造成模型虽然能够读取长上下文，却无法充分利用这些信息进行长距离推理。

## 相关工作：RoPE的一些变体

--YARN：在RoPE上做了两项改动：按频率分段的位置插值和注意力 logits 缩放。前者同时保留短距离位置分辨率与长距离外推能力，后者补偿上下文扩展后注意力分布的变化。

--FoPE：RoPE理论上依靠频率的周期性实现长度外推，但模型中的线性层、非线性激活以及有限训练长度会破坏这种理想频谱结构。为此，FoPE 不再假设每个注意力维度只包含一个频率，而是把每个维度建模为包含主频和若干附加频率的傅里叶级数，同时把未得到充分训练的极低频分量置为零，从而提高注意力周期延拓的鲁棒性和模型的长度泛化能力。

--AdaRoPE：不同注意力头不应该使用完全相同的RoPE旋转频率和缩放策略。因此，它把标准 RoPE 的全模型统一配置改为按注意力头自适应学习的频率和随上下文长度变化的缩放因子，以提高频率维度利用率。

--PoPE：把注意力中的内容匹配和相对位置匹配显式解耦，内容只决定特征幅值，位置只决定相位，避免内容任意改变一个注意力通道偏好的相对距离。

## 开源模型的情况

qwen3的技术报告[arXiv：2505.09388]中阐述：相较于qwen2.5,引入 YARN ，使推理期间的序列长度容量提升至原来的四倍；在 RULER 基准测试中，对qwen3采用 scaling factor=4 的 YARN ，在非思考模式下，其在长上下文处理任务上的表现优于规模相近的 Qwen2.5 模型。

kimi3的技术报告[arXiv:2607.24653]中阐述：kimi3采用了混合注意力，采用的其一注意力KDA，需要给每个token表示状态，状态按 token 顺序更新，输出依赖按顺序累积的状态，交换 token 顺序通常会产生不同结果，所以位置顺序被隐含在递归状态中。所以kimi3，没有显式加入位置编码的过程。

deepseek-v4的技术报告[arXiv:2606.19348]中阐述：deepseek-v4采用的是部分旋转位置编码，具体而言，对于CSA和HCA（模型采用的两种注意力）中使用的每个查询向量和 KV 条目向量，我们只对其最后 64 个维度应用RoPE

## 我的思考和疑问

考虑极限情况：如果上来就把context拉的很长，会出现什么现象，如果效果不好，可能的原因是什么？

[NeurIPS 2021 The Stability-Efficiency Dilemma]: 训练初期模型参数处于随机初始化附近，注意力分布近乎随机。长序列会引发梯度方差的极端值，直接对应训练 loss 的突然飙升，导致 Adam 优化器的二阶矩估计被噪声主导，优化过程本身崩溃。

[NeurIPS 2024  "Dataset Decomposition"]自然语言中绝大多数语法和语义关系发生在短距离内（通常 < 50 tokens），如主谓一致、修饰关系、代词指代等。如果训练数据中大量是长序列，模型在训练早期会被迫将大量计算资源花在处理的长距离关系上，而最需要优先学会的短距离模式反而被忽略。使用从短到长的渐进式方式，可以比固定长度基线快6倍达到目标精度，且下游任务性能更好。Meta 在 NAACL 2024 的消融实验也证实，在预训练数据集中拥有大量长文本并不是实现强长上下文性能的关键，少量高质量长文本配合持续预训练即可达到同等效果。

RoPE的变体在文章中有很好的实验效果，在一些开源模型中也会被使用，但是主流开源模型很少会花大篇幅去介绍为什么使用它（因此怀疑这个切入点是不是有意义？RoPE的各种变体是不是水文？）

## 原始Transformer的位置编码

原始 Transformer 使用固定的正弦—余弦绝对位置编码，并将其直接加到 token embedding 上。设序列位置为 $m$，模型隐藏维度为 $d_{\mathrm{model}}$，第 $i$ 组二维位置分量对应的频率为

$$
\theta_i=10000^{-\frac{2i}{d_{\mathrm{model}}}},
\qquad i=0,1,\ldots,\frac{d_{\mathrm{model}}}{2}-1.
$$

位置 $m$ 在第 $2i$ 和第 $2i+1$ 个维度上的编码分别为

$$
\mathrm{PE}(m,2i)=\sin(m\theta_i)
=\sin\left(\frac{m}{10000^{2i/d_{\mathrm{model}}}}\right),
$$

$$
\mathrm{PE}(m,2i+1)=\cos(m\theta_i)
=\cos\left(\frac{m}{10000^{2i/d_{\mathrm{model}}}}\right).
$$

因此，每一组二维位置分量可以写成

$$
\mathbf p_m^{(i)}=
\begin{bmatrix}
\sin(m\theta_i)\\
\cos(m\theta_i)
\end{bmatrix},
$$

完整的位置向量 $\mathbf p_m\in\mathbb R^{d_{\mathrm{model}}}$ 由所有频率分量拼接得到。对于位置 $m$ 处的 token embedding $\mathbf x_m$，输入 Transformer 的表示为

$$
\tilde{\mathbf x}_m=\mathbf x_m+\mathbf p_m.
$$

随后，内容信息和位置信息的和共同经过线性投影，生成 query、key 和 value：

$$
\mathbf q_m=W_Q\tilde{\mathbf x}_m,\qquad
\mathbf k_m=W_K\tilde{\mathbf x}_m,\qquad
\mathbf v_m=W_V\tilde{\mathbf x}_m.
$$

位置 $m$ 对位置 $n$ 的注意力 logit 为

$$
a_{m,n}=\frac{\mathbf q_m^{\mathsf T}\mathbf k_n}{\sqrt{d_k}},
$$

因此原始 Transformer 的位置编码属于**加性绝对位置编码**：位置信号在进入 Self-Attention 之前与内容表示相加，再通过 $W_Q$ 和 $W_K$ 共同影响注意力分数。

## RoPE 的位置编码

原始 Transformer 将位置编码与 token embedding 相加，位置信息会与内容信息混合后共同进入线性投影。RoPE 则采用另一种思路：**不直接修改 token 表示，而是根据 token 的位置旋转 query 和 key，使注意力内积自然包含相对位置信息。**

设位置 $m$ 处的 query 为 $\mathbf q_m$，位置 $n$ 处的 key 为 $\mathbf k_n$。RoPE 将它们的相邻两个维度组成一个二维分量 $\mathbf q_m^{(i)}$ 和 $\mathbf k_n^{(i)}$。第 $i$ 个二维分量对应频率

$$
\theta_i=b^{-\frac{2i}{d}},
\qquad i=0,1,\ldots,\frac d2-1,
$$

其中 $d$ 是 attention head 的维度，$b$ 是 RoPE base。定义二维旋转矩阵

$$
R(\phi)=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}.
$$

RoPE 根据绝对位置分别旋转 query 和 key：

$$
\tilde{\mathbf q}_m^{(i)}=R(m\theta_i)\mathbf q_m^{(i)},
\qquad
\tilde{\mathbf k}_n^{(i)}=R(n\theta_i)\mathbf k_n^{(i)}.
$$

关键在于，两个旋转后向量的内积满足

$$
\begin{aligned}
\left(\tilde{\mathbf q}_m^{(i)}\right)^{\mathsf T}
\tilde{\mathbf k}_n^{(i)}
&=\left(\mathbf q_m^{(i)}\right)^{\mathsf T}
R(m\theta_i)^{\mathsf T}R(n\theta_i)
\mathbf k_n^{(i)}\\
&=\left(\mathbf q_m^{(i)}\right)^{\mathsf T}
R\big((n-m)\theta_i\big)
\mathbf k_n^{(i)}.
\end{aligned}
$$

因此，虽然 query 和 key 分别按照绝对位置 $m$ 和 $n$ 进行旋转，但进入注意力分数的旋转角只与相对位置 $n-m$ 有关。完整的注意力 logit 为

$$
a_{m,n}^{\mathrm{RoPE}}
=\frac{1}{\sqrt d}
\sum_{i=0}^{d/2-1}
\left(\mathbf q_m^{(i)}\right)^{\mathsf T}
R\big((n-m)\theta_i\big)
\mathbf k_n^{(i)}.
$$

RoPE 的几何含义是：位置不会简单地增加一个额外向量，而是改变 query 和 key 在各个二维平面中的方向。两个 token 的相对距离决定它们之间额外的旋转角，从而改变向量对齐程度和注意力分数。不同维度使用不同频率，使模型能够同时表示细粒度的局部距离和变化较慢的长距离关系。

进一步将每个二维分量写成极坐标形式：

$$
\mathbf q_m^{(i)}
=\mu_{q,m}^{(i)}
\begin{bmatrix}
\cos\phi_{q,m}^{(i)}\\
\sin\phi_{q,m}^{(i)}
\end{bmatrix},
\qquad
\mathbf k_n^{(i)}
=\mu_{k,n}^{(i)}
\begin{bmatrix}
\cos\phi_{k,n}^{(i)}\\
\sin\phi_{k,n}^{(i)}
\end{bmatrix},
$$

则该分量对注意力分数的贡献为

$$
\mu_{q,m}^{(i)}\mu_{k,n}^{(i)}
\cos\left(
(n-m)\theta_i
+\phi_{k,n}^{(i)}
-\phi_{q,m}^{(i)}
\right).
$$

这个表达式揭示了 RoPE 的核心机制：

- $\mu_{q,m}^{(i)}\mu_{k,n}^{(i)}$ 主要反映两个 token 在该特征上的内容匹配强度；
- $(n-m)\theta_i$ 编码 query 与 key 的相对位置；
- $\phi_{k,n}^{(i)}-\phi_{q,m}^{(i)}$ 来自内容向量自身的方向。

因此，RoPE 确实把相对位置引入了注意力内积，但内容相位和位置相位最终出现在同一个余弦函数中。换言之，token 内容不仅决定匹配强度，也可能移动某个频率分量所偏好的相对位置。这种“内容—位置耦合”正是后续 PoPE 尝试消除的问题。
