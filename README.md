# How Can Positional Encoding Be Adjusted to Improve Length Extrapolation in Large Language Models?

**English** | [中文](README_CN.md) | [Experiments and Reproduction](EXPERIMENTS.md)

## Why It Matters

Many technical reports for open-source models discuss positional encoding (PE) and length extrapolation.

Better length extrapolation can substantially reduce the training cost of long-context models by enabling models to **train on short sequences and infer on long ones**.

It can unlock core long-text applications and directly improve the practical capabilities of large language models.

It can also deepen our theoretical understanding of how Transformers encode positional information and contribute to the evolution of model architectures.

## Introduction

**Length extrapolation** refers to the degradation in generalization that occurs when the input length at inference time differs from, and especially exceeds, the lengths observed during training. For example, if a model is trained only on sequences of 512 tokens, it may fail to process inputs longer than 512 tokens correctly. This limitation reduces its effectiveness on tasks such as long-document understanding and multi-turn dialogue.

**Why does positional encoding affect an LLM's length extrapolation?** The Transformer architecture cannot inherently perceive token order and therefore relies on positional encoding to inject sequence-order information. This creates a fundamental dependence on positional signals. During pretraining, however, the model observes positional-encoding patterns only within a limited context window. Its attention weights and feature representations are therefore learned for the positional distribution within this bounded interval. When the inference sequence exceeds the training range, the positional encoding may produce values or vectors that the model has never encountered. These unfamiliar positional signals fall outside the training distribution, potentially causing abnormal attention scores and a sharp decline in output quality. Because a positional encoding scheme determines how predictable its signals remain beyond the training length, it directly affects length extrapolation.

## Current Practice and Problem

RoPE applies the same scaling rule to every Transformer layer without distinguishing between shallow and deep layers. This uniform treatment cannot simultaneously satisfy the shallow layers' need for precise localization and the deep layers' need for weaker positional constraints. When the context length greatly exceeds the lengths seen during pretraining, the model struggles to preserve both local structural modeling and global semantic integration, ultimately reducing long-context retrieval or reasoning performance.

Uniform scaling creates two conflicting causal chains:

1. **Aggressive scaling:** Strongly compressing RoPE in every layer weakens fine-grained relative-position information in shallow layers. This can prevent shallow layers from correctly combining neighboring tokens, constructing local structures, and anchoring a long sequence to the semantic space learned during pretraining. The eventual result may be a sharp increase in perplexity and a collapse in long-context retrieval performance.
2. **Conservative scaling:** Preserving strong positional signals in every layer to protect shallow-layer behavior leaves deep layers subject to overly rigid positional constraints. Positional noise can then interfere with global semantic integration and compositional reasoning in deeper layers. The model may be able to read a long context without being able to use that information effectively for long-range reasoning.

## Related Work: RoPE Variants

- **YaRN:** YaRN makes two changes to RoPE: frequency-dependent position interpolation and attention-logit scaling. The former aims to preserve short-range positional resolution while enabling long-range extrapolation, whereas the latter compensates for changes in the attention distribution after context extension.

- **FoPE:** In theory, RoPE relies on frequency periodicity for length extrapolation, but linear layers, nonlinear activations, and finite training lengths disrupt this ideal spectral structure. FoPE therefore no longer assumes that each attention dimension contains only one frequency. Instead, it models every dimension as a Fourier series containing a principal frequency and several additional frequencies, while setting insufficiently trained, extremely low-frequency components to zero. This improves the robustness of periodic attention extension and the model's ability to generalize to longer sequences.

- **AdaRoPE:** Different attention heads should not necessarily use identical RoPE rotation frequencies and scaling strategies. AdaRoPE replaces the globally shared configuration of standard RoPE with head-wise adaptive frequencies and context-length-dependent scaling factors, improving utilization of the frequency dimensions.

- **PoPE:** PoPE explicitly decouples content matching from relative-position matching in attention. Content determines only feature magnitude, while position determines only phase, preventing token content from arbitrarily changing the relative distance preferred by an attention channel.

## Positional Encoding in Open-Source Models

The Qwen3 technical report [arXiv:2505.09388] states that, compared with Qwen2.5, Qwen3 introduces YaRN to increase the sequence-length capacity at inference time by a factor of four. On the RULER benchmark, Qwen3 with YaRN at a scaling factor of 4 outperforms similarly sized Qwen2.5 models on long-context tasks in non-thinking mode.

The Kimi3 technical report [arXiv:2607.24653] states that Kimi3 uses hybrid attention. One of its attention mechanisms, KDA, maintains a state representation for every token. The state is updated in token order, and each output depends on the sequentially accumulated state. Changing token order therefore generally changes the output, meaning that positional order is implicitly encoded in the recurrent state. Kimi3 consequently does not add an explicit positional encoding process.

The DeepSeek-V4 technical report [arXiv:2606.19348] states that DeepSeek-V4 uses partial rotary positional encoding. Specifically, for every query vector and KV entry vector used in CSA and HCA, the model applies RoPE only to the final 64 dimensions.

## Questions and Reflections

Consider an extreme setting: what happens if training begins immediately with a very long context window? If performance is poor, what are the possible causes?

**[NeurIPS 2021, “The Stability-Efficiency Dilemma”]** At the beginning of training, model parameters remain close to random initialization and the attention distribution is nearly random. Long sequences can produce extreme gradient variance, directly corresponding to sudden spikes in training loss. The second-moment estimates of the Adam optimizer may then become dominated by noise, causing the optimization process itself to collapse.

**[NeurIPS 2024, “Dataset Decomposition”]** Most syntactic and semantic relationships in natural language occur over short distances—typically fewer than 50 tokens—including subject–verb agreement, modification relations, and pronoun resolution. If the training data contains a large proportion of long sequences, the model may be forced to spend substantial computational resources on long-range relationships early in training, while neglecting the short-range patterns that it should learn first. A progressive short-to-long training schedule can reach the target accuracy up to six times faster than a fixed-length baseline while also improving downstream performance. Meta's ablation study at NAACL 2024 likewise indicates that a large quantity of long documents in the pretraining corpus is not essential for strong long-context performance; a small amount of high-quality long-form data combined with continued pretraining can achieve comparable results.

RoPE variants report strong experimental results and are also used in some open-source models, but mainstream open-source model reports rarely explain in detail why a particular variant was selected. This raises two questions: is this research direction practically meaningful, and are some RoPE variants merely incremental results with insufficient validation?

## Positional Encoding in the Original Transformer

The original Transformer uses fixed sinusoidal absolute positional encoding and adds it directly to the token embedding. Let the sequence position be $m$, the model hidden dimension be $d_{\mathrm{model}}$, and the frequency associated with the $i$-th two-dimensional positional component be

$$
\theta_i=10000^{-\frac{2i}{d_{\mathrm{model}}}},
\qquad i=0,1,\ldots,\frac{d_{\mathrm{model}}}{2}-1.
$$

The encodings of position $m$ in dimensions $2i$ and $2i+1$ are respectively

$$
\mathrm{PE}(m,2i)=\sin(m\theta_i)
=\sin\left(\frac{m}{10000^{2i/d_{\mathrm{model}}}}\right),
$$

$$
\mathrm{PE}(m,2i+1)=\cos(m\theta_i)
=\cos\left(\frac{m}{10000^{2i/d_{\mathrm{model}}}}\right).
$$

Each two-dimensional positional component can therefore be written as

$$
\mathbf p_m^{(i)}=
\begin{bmatrix}
\sin(m\theta_i)\\
\cos(m\theta_i)
\end{bmatrix},
$$

and the complete positional vector $\mathbf p_m\in\mathbb R^{d_{\mathrm{model}}}$ is formed by concatenating all frequency components. For the token embedding $\mathbf x_m$ at position $m$, the representation passed into the Transformer is

$$
\tilde{\mathbf x}_m=\mathbf x_m+\mathbf p_m.
$$

The sum of content and positional information is then projected linearly to produce the query, key, and value:

$$
\mathbf q_m=W_Q\tilde{\mathbf x}_m,\qquad
\mathbf k_m=W_K\tilde{\mathbf x}_m,\qquad
\mathbf v_m=W_V\tilde{\mathbf x}_m.
$$

The attention logit from position $m$ to position $n$ is

$$
a_{m,n}=\frac{\mathbf q_m^{\mathsf T}\mathbf k_n}{\sqrt{d_k}}.
$$

The original Transformer's positional encoding is therefore an **additive absolute positional encoding**: the positional signal is added to the content representation before Self-Attention and then affects the attention scores jointly through $W_Q$ and $W_K$.

## RoPE Positional Encoding

The original Transformer adds positional encoding to the token embedding, mixing position and content before linear projection. RoPE takes a different approach: **instead of directly modifying the token representation, it rotates queries and keys according to token position so that their attention inner product naturally contains relative-position information.**

Let the query at position $m$ be $\mathbf q_m$ and the key at position $n$ be $\mathbf k_n$. RoPE groups each pair of adjacent dimensions into two-dimensional components $\mathbf q_m^{(i)}$ and $\mathbf k_n^{(i)}$. The frequency associated with the $i$-th component is

$$
\theta_i=b^{-\frac{2i}{d}},
\qquad i=0,1,\ldots,\frac d2-1,
$$

where $d$ is the attention-head dimension and $b$ is the RoPE base. Define the two-dimensional rotation matrix

$$
R(\phi)=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}.
$$

RoPE rotates the query and key independently according to their absolute positions:

$$
\tilde{\mathbf q}_m^{(i)}=R(m\theta_i)\mathbf q_m^{(i)},
\qquad
\tilde{\mathbf k}_n^{(i)}=R(n\theta_i)\mathbf k_n^{(i)}.
$$

The key property is that the inner product of the two rotated vectors satisfies

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

Although the query and key are rotated according to the absolute positions $m$ and $n$, the rotation angle appearing in the attention score depends only on the relative position $n-m$. The complete attention logit is

$$
a_{m,n}^{\mathrm{RoPE}}
=\frac{1}{\sqrt d}
\sum_{i=0}^{d/2-1}
\left(\mathbf q_m^{(i)}\right)^{\mathsf T}
R\big((n-m)\theta_i\big)
\mathbf k_n^{(i)}.
$$

The geometric interpretation of RoPE is that position does not simply add an extra vector; instead, it changes the directions of the query and key in each two-dimensional plane. The relative distance between two tokens determines the additional rotation angle between them, thereby changing vector alignment and the attention score. Different dimensions use different frequencies, allowing the model to represent both fine-grained local distances and slowly varying long-range relationships.

To make the mechanism more explicit, write each two-dimensional component in polar form:

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
\end{bmatrix}.
$$

The contribution of this component to the attention score is then

$$
\mu_{q,m}^{(i)}\mu_{k,n}^{(i)}
\cos\left(
(n-m)\theta_i
+\phi_{k,n}^{(i)}
-\phi_{q,m}^{(i)}
\right).
$$

This expression reveals the central mechanism of RoPE:

- $\mu_{q,m}^{(i)}\mu_{k,n}^{(i)}$ primarily represents the strength of content matching between the two tokens for that feature.
- $(n-m)\theta_i$ encodes the relative position between the query and key.
- $\phi_{k,n}^{(i)}-\phi_{q,m}^{(i)}$ comes from the directions of the content vectors themselves.

RoPE therefore introduces relative position into the attention inner product, but the content phase and positional phase ultimately appear in the same cosine function. In other words, token content determines not only the matching strength but can also shift the relative distance preferred by a frequency component. This **content-position coupling** is precisely the issue that PoPE later attempts to remove.
