from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from configuration_riemann import RiemannConfig


class MultiHeadAttention(nn.Module):

    def __init__(self, config: RiemannConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_size
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_heads, self.head_size
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_heads, self.head_size
        ).transpose(1, 2)
        use_sdpa = hidden_states.device.type in ("cuda", "cpu")
        if use_sdpa:
            context_layer = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=attention_mask is None,
            )
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.head_size)
            if attention_mask is None:
                attention_mask = torch.triu(
                    torch.full(
                        (seq_length, seq_length),
                        torch.finfo(attention_scores.dtype).min,
                        dtype=attention_scores.dtype,
                        device=attention_scores.device,
                    ),
                    diagonal=1,
                )
            attention_scores = attention_scores + attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        context_layer = self.out(context_layer)
        return context_layer


class MLP(nn.Module):

    def __init__(self, config: RiemannConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):

    def __init__(self, config: RiemannConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        attention_layernorm_out = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            attention_layernorm_out, attention_mask=attention_mask
        )
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + hidden_states
        mlp_layernorm_out = self.post_attention_layernorm(attention_output)
        mlp_output = self.mlp(mlp_layernorm_out)
        layer_output = mlp_output + attention_output
        return layer_output


class RiemannPreTrainedModel(PreTrainedModel):
    config_class = RiemannConfig
    base_model_prefix = "riemann"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RiemannForCausalLM(RiemannPreTrainedModel, GenerationMixin):

    def __init__(self, config: RiemannConfig):
        super().__init__(config)
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _build_attention_mask(self, input_shape, attention_mask, dtype, device):
        batch_size, seq_length = input_shape
        causal = torch.triu(
            torch.full(
                (seq_length, seq_length),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=device,
            ),
            diagonal=1,
        )
        causal = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            padding = attention_mask[:, None, None, :].to(dtype=dtype)
            padding = (1.0 - padding) * torch.finfo(dtype).min
            return causal + padding
        return causal

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("pass either input_ids or inputs_embeds, not both")
        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("need either input_ids or inputs_embeds")
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=inputs_embeds.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        full_mask = (
            None
            if attention_mask is None
            else self._build_attention_mask(
                (batch_size, seq_length),
                attention_mask,
                hidden_states.dtype,
                hidden_states.device,
            )
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, full_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, full_mask)
        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, hidden_states=hidden_states
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
