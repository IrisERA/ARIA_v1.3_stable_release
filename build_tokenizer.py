"""
MK1 — Tokenizer Builder
========================
Builds a BPE tokenizer with 16000 vocab from all .txt files in data/
Saves to mk1/tokenizer.json

Run from ARIA folder:
    py -3.11 mk1/build_tokenizer.py
"""

import os
import json
import re
from collections import Counter, defaultdict
from datetime import datetime

VOCAB_SIZE   = 16000
DATA_DIR     = "data"
OUTPUT_PATH  = "mk1/tokenizer.json"
SPECIAL      = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

def get_word_freqs(text):
    """Split text into words and count frequencies."""
    words = re.findall(r"\s?\S+", text)
    freq  = Counter()
    for w in words:
        # Represent word as space-separated chars with end marker
        chars = " ".join(list(w))
        freq[chars] += 1
    return freq

def get_pairs(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram    = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def build_bpe(text, vocab_size):
    print(f"Building BPE tokenizer — target vocab: {vocab_size:,}")
    
    word_freqs = get_word_freqs(text)
    print(f"Unique words: {len(word_freqs):,}")
    
    # Start with character vocab
    vocab = set()
    for word in word_freqs:
        for char in word.split():
            vocab.add(char)
    
    print(f"Initial char vocab: {len(vocab):,}")
    
    merges = []
    bpe_vocab = dict(word_freqs)
    
    target_merges = vocab_size - len(SPECIAL) - len(vocab)
    print(f"Need {target_merges:,} merges")
    
    for i in range(target_merges):
        pairs = get_pairs(bpe_vocab)
        if not pairs:
            break
        
        best = max(pairs, key=pairs.get)
        merges.append(list(best))
        bpe_vocab = merge_vocab(best, bpe_vocab)
        vocab.add("".join(best))
        
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,} merges done... vocab size: {len(vocab) + len(SPECIAL):,}")
    
    return vocab, merges

def main():
    # Load all text files
    print(f"\nLoading data from {DATA_DIR}/")
    all_text = ""
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".txt"):
            fpath = os.path.join(DATA_DIR, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            all_text += text + "\n"
            print(f"  {fname}: {len(text):,} chars")
    
    print(f"\nTotal: {len(all_text):,} chars")
    
    # Build BPE
    vocab_set, merges = build_bpe(all_text, VOCAB_SIZE)
    
    # Build final vocab dict
    vocab = {}
    for i, special in enumerate(SPECIAL):
        vocab[special] = i
    
    for token in sorted(vocab_set):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"\nFinal vocab size: {len(vocab):,}")
    print(f"Total merges: {len(merges):,}")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": merges}, f, ensure_ascii=False, indent=2)
    
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"\nSaved: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"\nNow update MK1Config:")
    print(f"  vocab_size: int = {len(vocab)}")
    print(f"\nThen delete old model files and retrain from scratch.")

if __name__ == "__main__":
    main()
