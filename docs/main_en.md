# How Can Positional Encoding Be Adjusted to Improve Length Extrapolation in Large Language Models?

## Value

Substantially reduce the training cost of long-context models and enable models to **train on short sequences and infer on long ones**.

Unlock core long-text applications and directly improve the practical capabilities of large language models.

Deepen our theoretical understanding of how Transformers encode positional information and contribute to the evolution of model architectures.

## Introduction

**Length extrapolation** refers to the degradation in generalization that occurs when the input length at inference time exceeds the lengths observed during training. For example, if a model is trained only on sequences of up to 512 tokens, it may fail to process inputs longer than 512 tokens correctly. This limitation reduces its effectiveness on tasks such as long-document understanding and multi-turn dialogue.

**Why does positional encoding affect length extrapolation?** The Transformer architecture cannot inherently perceive token order and therefore relies on positional encoding to inject sequence-order information. As a result, the model develops a fundamental dependence on positional signals. During pretraining, however, it observes positional-encoding patterns only within a limited context window. Its attention weights and feature representations are therefore optimized for the positional distribution within this bounded interval. When the inference sequence exceeds the training length, the positional encoding may produce values or vectors that the model has never encountered. These unfamiliar positional signals are out of distribution and can lead to abnormal attention scores and a sharp decline in output quality. Consequently, the predictability of a positional encoding scheme beyond the training range directly affects the model’s ability to extrapolate to longer contexts.

## Current Practice and Limitations

Mainstream open-source models, such as GLM and DeepSeek, use RoPE and typically apply the same RoPE scaling rule to every Transformer layer, without distinguishing between shallow and deep layers. This uniform treatment cannot simultaneously satisfy the shallow layers’ need for precise positional localization and the deep layers’ need for weaker positional constraints. As a result, it is difficult to balance local structural modeling with global semantic integration, ultimately degrading long-context retrieval and reasoning performance.

Uniform scaling creates two conflicting causal chains:

1. **Aggressive scaling:** Strongly compressing RoPE across all layers weakens fine-grained relative-position information in shallow layers. These layers may then fail to combine neighboring tokens correctly, construct local structures, and anchor long sequences to the semantic space learned during pretraining. This can cause perplexity to increase sharply and long-context retrieval performance to collapse.
2. **Conservative scaling:** Preserving strong positional signals across all layers to protect shallow-layer behavior leaves deep layers subject to overly rigid positional constraints. Positional noise can then interfere with global semantic integration and compositional reasoning. The model may be able to read a long context but still fail to use the information effectively for long-range reasoning.

## Related Work: RoPE Variants

- **YaRN:** YaRN makes two main changes to RoPE: frequency-dependent position interpolation and attention-logit scaling. The former aims to preserve short-range positional resolution while enabling long-range extrapolation, whereas the latter compensates for changes in the attention distribution after the context window is extended.

- **FoPE:** In theory, RoPE relies on frequency periodicity to extrapolate beyond the training length. In practice, linear layers, nonlinear activations, and finite training contexts disrupt this ideal spectral structure. FoPE therefore no longer assumes that each attention dimension contains only one frequency. Instead, it models each dimension as a Fourier series containing a dominant frequency and several auxiliary frequencies. It also removes extremely low-frequency components that are insufficiently trained, improving the robustness of periodic attention extension and the model’s length-generalization ability.

- **AdaRoPE:** AdaRoPE argues that different attention heads should not use identical RoPE rotation frequencies and scaling strategies. It replaces the globally shared RoPE configuration with head-wise learnable frequencies and context-length-dependent scaling factors, thereby improving the utilization of frequency dimensions.

- **PoPE:** PoPE explicitly decouples content matching—the **what**—from relative-position matching—the **where**. Content determines only the feature magnitude, while position determines only the phase, preventing token content from arbitrarily shifting the relative distance preferred by an attention channel.

## Questions and Research Directions

Consider an extreme setting: what happens if training begins immediately with a very long context window? If performance is poor, what are the underlying causes?

**[NeurIPS 2021, “The Stability-Efficiency Dilemma”]** At the beginning of training, model parameters remain close to their random initialization and the attention distribution is nearly random. Long sequences can produce extreme gradient variance, directly corresponding to sudden spikes in training loss. The second-moment estimates of the Adam optimizer may then become dominated by noise, potentially destabilizing or even collapsing the optimization process.

**[NeurIPS 2024, “Dataset Decomposition”]** Most syntactic and semantic relationships in natural language occur over short distances—typically fewer than 50 tokens—including subject–verb agreement, modification relations, and pronoun resolution. If the training data contains a large proportion of long sequences, the model may be forced to spend substantial computation on long-range relationships early in training, while neglecting the short-range patterns that should be learned first. A progressive short-to-long training schedule can reach the target accuracy up to six times faster than a fixed-length baseline while also improving downstream performance. Meta’s ablation study at NAACL 2024 likewise suggests that a large volume of long documents in the pretraining corpus is not essential for strong long-context performance; a relatively small amount of high-quality long-form data combined with continued pretraining may achieve comparable results.

Given the strong experimental performance reported for many RoPE variants, why have they not been widely adopted by mainstream open-source models? Does this indicate that the research direction has limited practical value, or are the reported improvements overly incremental or insufficiently validated?
