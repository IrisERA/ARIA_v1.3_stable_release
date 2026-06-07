"""
MK1 — Transformer Model
========================
The actual brain. Built from scratch using only NumPy.
No PyTorch, no TensorFlow, no frameworks.

Architecture:
    Token Embedding      → converts token IDs to vectors
    Positional Encoding  → tells model where each token is
    Attention Heads (4)  → model focuses on relevant context
    Transformer Layers   → stacked attention + feedforward
    Output Head          → predicts next token probabilities

This is a decoder-only transformer — same family as GPT.
"""

import numpy as np
import json
import os
import time
from typing import Optional


# ── Hyperparameters ────────────────────────────────────────────────────────
class MK1Config:
    """
    MK1 model configuration.
    These numbers control the size and behaviour of the model.
    Tweak these to make it bigger or smaller.
    """
    vocab_size:   int   = 1000    # must match tokenizer vocab size
    context_len:  int   = 128     # how many tokens the model sees at once
    embed_dim:    int   = 128     # size of each token's vector representation
    num_heads:    int   = 4       # number of attention heads
    num_layers:   int   = 4       # number of transformer blocks stacked
    ff_dim:       int   = 512     # feedforward hidden layer size (usually 4x embed_dim)
    dropout:      float = 0.1     # randomly zero out activations during training
    lr:           float = 3e-4    # learning rate — how fast weights update


# ── Math primitives ────────────────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax: converts raw scores to probabilities that sum to 1."""
    x = x - x.max(axis=axis, keepdims=True)  # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation — smoother than ReLU, used in GPT."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalise activations — keeps training stable."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# ── Weight initialisation ──────────────────────────────────────────────────

def xavier(shape: tuple) -> np.ndarray:
    """Xavier initialisation — prevents vanishing/exploding gradients."""
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def zeros(shape: tuple) -> np.ndarray:
    return np.zeros(shape)


def ones(shape: tuple) -> np.ndarray:
    return np.ones(shape)


# ── Positional Encoding ────────────────────────────────────────────────────

def positional_encoding(context_len: int, embed_dim: int) -> np.ndarray:
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.
    Gives the model a sense of where each token is in the sequence.

    Returns shape: (context_len, embed_dim)
    """
    pe  = np.zeros((context_len, embed_dim))
    pos = np.arange(context_len)[:, None]          # (T, 1)
    div = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000) / embed_dim))

    pe[:, 0::2] = np.sin(pos * div)   # even dims → sin
    pe[:, 1::2] = np.cos(pos * div)   # odd dims  → cos
    return pe


# ── Attention ──────────────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    The core mechanism of the transformer.

    Each head learns to attend to different aspects of the input.
    Head 1 might focus on nearby words.
    Head 2 might focus on grammatical relationships.
    Head 3 might focus on semantic meaning.
    etc.

    Weights:
        Wq, Wk, Wv → project input into Q, K, V spaces
        Wo          → project concatenated heads back to embed_dim
    """

    def __init__(self, cfg: MK1Config):
        self.cfg       = cfg
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.embed_dim // cfg.num_heads
        D = cfg.embed_dim

        # Query, Key, Value projection weights
        self.Wq = xavier((D, D))
        self.Wk = xavier((D, D))
        self.Wv = xavier((D, D))
        self.Wo = xavier((D, D))

        # Biases
        self.bq = zeros((D,))
        self.bk = zeros((D,))
        self.bv = zeros((D,))
        self.bo = zeros((D,))

        # Store for backprop
        self._cache = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        x shape: (batch, seq_len, embed_dim)
        returns: (batch, seq_len, embed_dim)
        """
        B, T, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # Project to Q, K, V
        Q = x @ self.Wq + self.bq   # (B, T, D)
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        # Split into heads: (B, H, T, Hd)
        Q = Q.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

        # Scaled dot product attention
        scale = np.sqrt(Hd)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale   # (B, H, T, T)

        # Causal mask — model can only see past tokens, not future
        # This is what makes it a language model (predicts next token)
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores[:, :, mask] = -1e9   # future positions → -inf → softmax → 0

        attn = softmax(scores)       # (B, H, T, T)

        # Attend to values
        out = attn @ V               # (B, H, T, Hd)

        # Merge heads back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = out @ self.Wo + self.bo

        # Cache for backprop
        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "attn": attn, "B": B, "T": T}

        return out

    def parameters(self) -> list:
        return [self.Wq, self.Wk, self.Wv, self.Wo,
                self.bq, self.bk, self.bv, self.bo]


# ── Feed Forward ───────────────────────────────────────────────────────────

class FeedForward:
    """
    Two-layer MLP applied to each position independently.
    Expands to ff_dim then contracts back to embed_dim.
    This is where the model stores most of its "knowledge".
    """

    def __init__(self, cfg: MK1Config):
        D  = cfg.embed_dim
        FF = cfg.ff_dim

        self.W1 = xavier((D, FF))
        self.b1 = zeros((FF,))
        self.W2 = xavier((FF, D))
        self.b2 = zeros((D,))

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache["x"] = x
        h = gelu(x @ self.W1 + self.b1)
        self._cache["h"] = h
        return h @ self.W2 + self.b2

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]


# ── Transformer Block ──────────────────────────────────────────────────────

class TransformerBlock:
    """
    One complete transformer layer:
        LayerNorm → Attention → residual connection
        LayerNorm → FeedForward → residual connection

    Residual connections (the + x parts) are critical —
    they let gradients flow back through many layers without vanishing.
    """

    def __init__(self, cfg: MK1Config):
        D = cfg.embed_dim

        self.attn = MultiHeadAttention(cfg)
        self.ff   = FeedForward(cfg)

        # LayerNorm parameters (learned)
        self.ln1_g = ones((D,))
        self.ln1_b = zeros((D,))
        self.ln2_g = ones((D,))
        self.ln2_b = zeros((D,))

        self._cache = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        # Attention sublayer with residual
        normed = layer_norm(x, self.ln1_g, self.ln1_b)
        x = x + self.attn.forward(normed, training)

        # Feedforward sublayer with residual
        normed = layer_norm(x, self.ln2_g, self.ln2_b)
        x = x + self.ff.forward(normed)

        self._cache["out"] = x
        return x

    def parameters(self) -> list:
        return (self.attn.parameters() +
                self.ff.parameters() +
                [self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b])


# ── MK1 Model ──────────────────────────────────────────────────────────────

class MK1Model:
    """
    The complete MK1 transformer language model.

    Forward pass:
        token IDs → embeddings → + positional encoding
        → N transformer blocks
        → layer norm
        → linear projection → logits over vocab
        → softmax → probabilities

    This is a decoder-only GPT-style model.
    """

    def __init__(self, cfg: MK1Config):
        self.cfg = cfg
        V = cfg.vocab_size
        D = cfg.embed_dim
        T = cfg.context_len

        # Token embedding table: each token ID maps to a D-dim vector
        self.token_embed = xavier((V, D)) * 0.02

        # Positional encoding (fixed, not learned)
        self.pos_enc = positional_encoding(T, D)

        # Stack of transformer blocks
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.num_layers)]

        # Final layer norm
        self.ln_f_g = ones((D,))
        self.ln_f_b = zeros((D,))

        # Output projection: D → vocab_size (predicts next token)
        self.head_w = xavier((D, V)) * 0.02

        # Parameter count
        self._count_params()

    def forward(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Args:
            token_ids: (batch, seq_len) integer array
        Returns:
            logits: (batch, seq_len, vocab_size) — raw scores for each token
        """
        B, T = token_ids.shape
        assert T <= self.cfg.context_len, f"Sequence too long: {T} > {self.cfg.context_len}"

        # Embed tokens + add positional encoding
        x = self.token_embed[token_ids]    # (B, T, D)
        x = x + self.pos_enc[:T]           # broadcast position info

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, training)

        # Final normalisation
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)

        # Project to vocab size
        logits = x @ self.head_w           # (B, T, V)

        return logits

    def loss(self, token_ids: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss for next-token prediction.

        Input:  token_ids (B, T)
        Target: token_ids shifted by 1 (predict next token)
        """
        B, T = token_ids.shape

        # inputs = all tokens except last
        # targets = all tokens except first (shifted by 1)
        inputs  = token_ids[:, :-1]   # (B, T-1)
        targets = token_ids[:, 1:]    # (B, T-1)

        logits = self.forward(inputs, training=True)   # (B, T-1, V)

        # Cross-entropy loss
        B, T, V = logits.shape
        logits_flat  = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        # Softmax + log for numerical stability
        log_probs = logits_flat - np.log(
            np.exp(logits_flat - logits_flat.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)
        ) - logits_flat.max(axis=-1, keepdims=True)

        # Pick log prob of correct token
        loss = -log_probs[np.arange(B * T), targets_flat].mean()

        return loss, logits

    @np.errstate(all='ignore')
    def generate(self, start_ids: list[int], max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40) -> list[int]:
        """
        Generate new tokens autoregressively.

        temperature: higher = more creative/random, lower = more focused
        top_k:       only sample from the top K most likely tokens
        """
        ids = list(start_ids)

        for _ in range(max_new_tokens):
            # Crop to context length
            context = ids[-self.cfg.context_len:]
            x = np.array([context])   # (1, T)

            # Forward pass
            logits = self.forward(x)   # (1, T, V)
            next_logits = logits[0, -1, :]   # last position (1, V)

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_indices = np.argsort(next_logits)[-top_k:]
                mask = np.ones(len(next_logits), dtype=bool)
                mask[top_k_indices] = False
                next_logits[mask] = -1e9

            # Sample from distribution
            probs = softmax(next_logits)
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(int(next_id))

        return ids

    def _count_params(self):
        total = 0
        total += self.token_embed.size
        for block in self.blocks:
            for p in block.parameters():
                total += p.size
        total += self.head_w.size
        self.num_params = total
        print(f"MK1 Model initialised — {total:,} parameters")
        print(f"  Layers: {self.cfg.num_layers} | Heads: {self.cfg.num_heads} | Embed: {self.cfg.embed_dim}")

    def save(self, path: str):
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        weights = {
            "token_embed": self.token_embed,
            "pos_enc":     self.pos_enc,
            "ln_f_g":      self.ln_f_g,
            "ln_f_b":      self.ln_f_b,
            "head_w":      self.head_w,
        }
        for i, block in enumerate(self.blocks):
            params = block.parameters()
            names  = ["Wq","Wk","Wv","Wo","bq","bk","bv","bo",
                      "W1","b1","W2","b2","ln1_g","ln1_b","ln2_g","ln2_b"]
            for name, param in zip(names, params):
                weights[f"block{i}_{name}"] = param

        np.savez_compressed(path, **weights)
        size = os.path.getsize(path + ".npz") / 1e6
        print(f"Model saved: {path}.npz ({size:.1f} MB)")

    def load(self, path: str):
        """Load model weights from disk."""
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        self.token_embed = data["token_embed"]
        self.pos_enc     = data["pos_enc"]
        self.ln_f_g      = data["ln_f_g"]
        self.ln_f_b      = data["ln_f_b"]
        self.head_w      = data["head_w"]

        names = ["Wq","Wk","Wv","Wo","bq","bk","bv","bo",
                 "W1","b1","W2","b2","ln1_g","ln1_b","ln2_g","ln2_b"]
        for i, block in enumerate(self.blocks):
            params = block.parameters()
            for name, param in zip(names, params):
                key = f"block{i}_{name}"
                if key in data:
                    param[:] = data[key]

        print(f"Model loaded: {path}")


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  MK1 Transformer — Architecture Test")
    print("=" * 50)

    cfg   = MK1Config()
    model = MK1Model(cfg)

    print(f"\nRunning forward pass test...")
    # Fake batch: 2 sequences of 16 tokens
    test_ids = np.random.randint(0, cfg.vocab_size, (2, 16))
    logits   = model.forward(test_ids)
    print(f"Input shape:  {test_ids.shape}")
    print(f"Output shape: {logits.shape}  (batch, seq, vocab)")

    print(f"\nRunning loss test...")
    test_ids = np.random.randint(0, cfg.vocab_size, (2, 32))
    loss, _  = model.loss(test_ids)
    print(f"Initial loss: {loss:.4f}  (expected ~{np.log(cfg.vocab_size):.2f} for random weights)")

    print(f"\nRunning generation test...")
    start = [4, 19, 57]   # 3 random token IDs
    generated = model.generate(start, max_new_tokens=20, temperature=0.8)
    print(f"Generated {len(generated)} tokens: {generated}")

    print(f"\nSaving model...")
    model.save("mk1/mk1_model")

    print(f"\nMK1 architecture is working.")
    print(f"Next step: train it on Shakespeare.")
