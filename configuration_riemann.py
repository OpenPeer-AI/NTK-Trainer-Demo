from __future__ import annotations
from transformers import PretrainedConfig


class RiemannConfig(PretrainedConfig):
    model_type = "riemann_openpeer"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 1024,
        layer_norm_eps: float = 1e-05,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
