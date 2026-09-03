from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.block_size <= 0 or self.vocab_size <= 0:
            raise ValueError("block_size and vocab_size must be positive")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(inputs.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype=inputs.dtype) * self.weight.to(dtype=inputs.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        positions = torch.arange(sequence_length, device=query.device, dtype=torch.float32)
        inverse_frequency = self.get_buffer("inverse_frequency")
        angles = torch.outer(positions, inverse_frequency.float())
        cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]
        return self._rotate(query, cosine, sine), self._rotate(key, cosine, sine)

    @staticmethod
    def _rotate(inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        )
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.query_key_value = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.output_projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_base)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = inputs.shape
        query_key_value = self.query_key_value(inputs)
        query, key, value = query_key_value.split(embedding_size, dim=-1)

        query = query.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        query, key = self.rotary(query, key)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        return self.residual_dropout(self.output_projection(attended))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.input_projection = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.output_projection = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_projection(inputs)
        inputs = F.gelu(inputs, approximate="tanh")
        return self.dropout(self.output_projection(inputs))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))


class RoPETransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)
        residual_standard_deviation = 0.02 / math.sqrt(2 * config.n_layer)
        for module in self.blocks:
            block = cast(TransformerBlock, module)
            nn.init.normal_(block.attention.output_projection.weight, mean=0.0, std=residual_standard_deviation)
            nn.init.normal_(block.mlp.output_projection.weight, mean=0.0, std=residual_standard_deviation)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.block_size:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds block size {self.config.block_size}"
            )

        hidden_states = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def configure_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        trainable = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        decay = [parameter for parameter in trainable.values() if parameter.dim() >= 2]
        no_decay = [parameter for parameter in trainable.values() if parameter.dim() < 2]
        parameter_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": betas,
        }
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            optimizer_kwargs["fused"] = device_type == "cuda"
        return torch.optim.AdamW(parameter_groups, **optimizer_kwargs)

    def parameter_count(self, non_embedding: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.token_embedding.weight.numel()
        return count

    def config_dict(self) -> dict:
        return asdict(self.config)
