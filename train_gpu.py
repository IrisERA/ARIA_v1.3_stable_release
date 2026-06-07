"""
MK1 — GPU Accelerated Trainer (DirectML compatible)
=====================================================
Rebuilt to avoid TransformerDecoder issues with DirectML.
Uses manual attention implementation that works on AMD via DirectML.

Run with Python 3.11:
    py -3.11 mk1/train_gpu.py --branch H --epochs 10 --steps 1000
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch_directml
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import argparse
from datetime import datetime

# Device
dml = torch_directml.device()
print(f"Device: {dml}")
                                 
                                 #--------------------------]
                                 #                          -]
class MK1Config:                 # To increase to 1B --     --]
    vocab_size:  int   = 1000    #                          ---]
    context_len: int   = 128     # 512    #set to           ----]
    embed_dim:   int   = 1024    # 2048   #300M parameters  -----]
    num_heads:   int   = 16      # 16     #16 -32           ------]
    num_layers:  int   = 24      # 24     #token mem--      -------]
    ff_dim:      int   = 4096    # 8192                     --------]
    dropout:     float = 0.1     #                          ---------]
    lr:          float = 3e-4    #------------------------------------]
                    

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.embed_dim // cfg.num_heads
        self.embed_dim = cfg.embed_dim

        self.qkv  = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, Hd   = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale  = Hd ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale   # (B, H, T, T)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.embed_dim)
        self.ff   = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MK1GPU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg         = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed   = nn.Embedding(cfg.context_len, cfg.embed_dim)
        self.drop        = nn.Dropout(cfg.dropout)
        self.blocks      = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f        = nn.LayerNorm(cfg.embed_dim)
        self.head        = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        total = sum(p.numel() for p in self.parameters())
        print(f"MK1-GPU ready — {total:,} parameters")

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.token_embed(idx) + self.pos_embed(pos))
        x    = self.blocks(x)
        x    = self.ln_f(x)
        return self.head(x)

    def generate(self, start_ids, max_new=100, temperature=0.8, top_k=40):
        self.eval()
        ids = torch.tensor([start_ids], dtype=torch.long, device=dml)
        with torch.no_grad():
            for _ in range(max_new):
                ctx    = ids[:, -self.cfg.context_len:]
                logits = self(ctx)[:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, -1:]] = -float('inf')
                probs   = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                ids     = torch.cat([ids, next_id], dim=1)
        return ids[0].tolist()

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path + ".pt")
        print(f"Saved: {path}.pt ({os.path.getsize(path+'.pt')/1e6:.1f} MB)")

    def load(self, path):
        if not path.endswith(".pt"): path += ".pt"
        self.load_state_dict(torch.load(path, map_location=dml))
        print(f"Loaded: {path}")


def load_tokenizer(path="mk1/tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab   = data["vocab"]
    vocab_r = {v: k for k, v in vocab.items()}
    merges  = [tuple(m) for m in data["merges"]]
    return vocab, vocab_r, merges


def encode(text, vocab, merges):
    import re
    unk  = vocab.get("<UNK>", 0)
    ids  = []
    for word in re.findall(r"\s?\S+", text):
        toks = list(word)
        for pair in merges:
            i, new = 0, []
            while i < len(toks):
                if i < len(toks)-1 and toks[i]==pair[0] and toks[i+1]==pair[1]:
                    new.append(pair[0]+pair[1]); i += 2
                else:
                    new.append(toks[i]); i += 1
            toks = new
        ids.extend(vocab.get(t, unk) for t in toks)
    return ids


def decode(ids, vocab_r):
    special = {"<PAD>","<UNK>","<BOS>","<EOS>"}
    return "".join(vocab_r.get(i,"?") for i in ids if vocab_r.get(i,"?") not in special)


def get_batch(data, batch_size, context_len):
    ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    x  = torch.stack([data[i:i+context_len]   for i in ix]).to(dml)
    y  = torch.stack([data[i+1:i+context_len+1] for i in ix]).to(dml)
    return x, y


def train(branch="H", epochs=10, steps_per_epoch=1000, data_path="data/shakespeare.txt", lr=3e-4, sample_prompt="To be"):
    print("=" * 55)
    print(f"  MK1-{branch} | GPU Training | DirectML")
    print("=" * 55)

    vocab, vocab_r, merges = load_tokenizer()
    cfg = MK1Config()
    cfg.vocab_size = len(vocab)

    model_path = f"mk1/mk1{branch}_gpu_model"
    model = MK1GPU(cfg).to(dml)

    if os.path.exists(model_path + ".pt"):
        model.load(model_path)

    # Load ALL .txt files in data/ folder automatically
    data_dir = "data"
    txt_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ]) if os.path.isdir(data_dir) else [data_path]

    print(f"\nLoading {len(txt_files)} data file(s)...")
    all_ids = []
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = encode(text, vocab, merges)
        all_ids.extend(ids)
        print(f"  {os.path.basename(fpath)} — {len(ids):,} tokens")

    print(f"  Total: {len(all_ids):,} tokens")
    data   = torch.tensor(all_ids, dtype=torch.long)
    split  = int(len(data) * 0.9)
    train_data = data[:split]
    val_data   = data[split:]
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Learning rate: {lr}")
    best_val = float("inf")
    os.makedirs("mk1/logs", exist_ok=True)
    log_path = f"mk1/logs/mk1_{branch}_gpu.log"

    print(f"\nTraining {epochs} × {steps_per_epoch} = {epochs*steps_per_epoch:,} steps\n")

    for epoch in range(1, epochs + 1):
        print(f"\n-- Epoch {epoch}/{epochs} --------------------------")
        model.train()

        for step in range(1, steps_per_epoch + 1):
            gs = (epoch-1)*steps_per_epoch + step
            x, y = get_batch(train_data, 8, cfg.context_len)

            logits = model(x)
            loss   = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if gs % 10 == 0:
                t    = datetime.now().strftime("%H:%M:%S")
                line = f"{t} | step {gs:6d} | loss {loss.item():.4f}"

                if gs % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        vx, vy = get_batch(val_data, 32, cfg.context_len)
                        vl = F.cross_entropy(model(vx).reshape(-1,cfg.vocab_size), vy.reshape(-1)).item()
                    line += f" | val {vl:.4f}"
                    if vl < best_val:
                        best_val = vl
                        model.save(model_path + "_best")
                    model.train()

                print(line)
                with open(log_path, "a") as f:
                    f.write(line + "\n")

            if gs % 500 == 0:
                model.save(model_path)

        # Sample
        model.eval()
        start = encode(sample_prompt, vocab, merges)
        gen   = model.generate(start, max_new=100, temperature=0.8)
        print(f"\n  Sample: '{decode(gen, vocab_r)[:200]}'")

    model.save(model_path)
    print(f"\nDone! Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["H","A"], default="H")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps",  type=int, default=1000)
    parser.add_argument("--data",   type=str, default="data/shakespeare.txt")
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--sample", type=str,   default="To be")
    args = parser.parse_args()
    train(args.branch, args.epochs, args.steps, args.data, args.lr, args.sample)
