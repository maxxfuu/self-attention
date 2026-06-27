"""GPT-124M architecture.

A from-scratch implementation of the GPT-2 (124M) transformer, assembled from:
token + positional embeddings -> N transformer blocks -> final LayerNorm -> output head.

Each transformer block uses pre-LayerNorm, multi-head causal self-attention,
a GELU feed-forward network, and residual (shortcut) connections.

This module is the shared source of truth imported by the notebooks
(e.g. ``from gpt_architecture import GPTModel``).
"""

import math

import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Max context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of transformer blocks
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,      # Bias for the Q/K/V projections
}


class LayerNorm(nn.Module):
    """Normalizes across the embedding features, then applies a learnable scale and shift."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Small constant to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation using the tanh approximation from the GPT-2 paper."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Position-wise MLP that expands the embedding dimension by 4x, applies GELU, and projects back down."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention implemented as a single class with weight splits."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0, "emb_dim must be divisible by n_heads"

        self.d_out = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.d_out // self.n_heads

        self.W_q = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # Causal mask: upper-triangular (excluding diagonal) is masked out
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(cfg["context_length"], cfg["context_length"]),
                diagonal=1,
            ).bool(),
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # (batch, num_tokens, d_out) -> (batch, n_heads, num_tokens, head_dim)
        def split_heads(t):
            return t.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(queries)
        keys = split_heads(keys)
        values = split_heads(values)

        # Scaled dot-product attention scores per head
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply the causal mask
        mask = self.mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, then merge heads back together
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections around both sub-layers."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sub-layer with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward sub-layer with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    """Token + positional embeddings, a stack of transformer blocks, a final LayerNorm, and a linear head."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy autoregressive generation: appends the argmax token one step at a time."""
    for _ in range(max_new_tokens):
        # Crop the context to the supported length
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on the last time step, then pick the most likely next token
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
