"""
MK1 — Tokenizer
================
Converts raw text into numbers the model can process.
And converts numbers back into text for output.

We're building BPE (Byte Pair Encoding) from scratch.
This is the same algorithm used in GPT — no libraries.

How BPE works:
    1. Start with every character as its own token
    2. Find the most common pair of tokens
    3. Merge them into a new token
    4. Repeat until vocab is the size you want

Example:
    "hello world" 
    → characters: ['h','e','l','l','o',' ','w','o','r','l','d']
    → after merges: ['hel', 'lo', ' ', 'wor', 'ld']
    → numbers: [42, 17, 3, 89, 56]
"""

import os
import json
import re
from collections import Counter, defaultdict
from typing import Optional


class Tokenizer:
    """
    BPE Tokenizer built from scratch.

    Usage:
        tok = Tokenizer()
        tok.train("path/to/text.txt", vocab_size=1000)
        tok.save("mk1_tokenizer.json")

        # Later:
        tok = Tokenizer.load("mk1_tokenizer.json")
        ids = tok.encode("hello world")
        text = tok.decode(ids)
    """

    # Special tokens every model needs
    PAD   = "<PAD>"    # padding — fills empty space in batches
    UNK   = "<UNK>"    # unknown token — things we've never seen
    BOS   = "<BOS>"    # beginning of sequence
    EOS   = "<EOS>"    # end of sequence
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self):
        self.vocab:        dict[str, int] = {}   # token → id
        self.vocab_r:      dict[int, str] = {}   # id → token (reverse)
        self.merges:       list[tuple]    = []   # BPE merge rules
        self.trained:      bool           = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text_source: str, vocab_size: int = 2000, verbose: bool = True):
        """
        Train the tokenizer on a text file or raw string.

        Args:
            text_source: path to .txt file OR raw text string
            vocab_size:  how many tokens in vocabulary
                         256  = character level (tiny, slow generation)
                         1000 = good for small datasets like Shakespeare
                         8000 = GPT-2 level (needs much more data)
            verbose:     print progress
        """

        # Load text
        if os.path.exists(text_source):
            with open(text_source, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            if verbose:
                print(f"Loaded: {text_source} ({len(text):,} chars)")
        else:
            text = text_source
            if verbose:
                print(f"Training on raw text ({len(text):,} chars)")

        # Step 1: Build initial character vocab
        if verbose:
            print("Building character vocabulary...")

        chars = sorted(set(text))
        self.vocab = {}

        # Special tokens get the first IDs
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.vocab[tok] = i

        # Then all characters
        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        if verbose:
            print(f"  Base vocab: {len(self.vocab)} tokens ({len(chars)} chars + {len(self.SPECIAL_TOKENS)} special)")

        # Step 2: Tokenize text into characters
        # Represent as list of tuples of characters per word
        words = self._get_word_freqs(text)

        # Step 3: BPE merge loop
        num_merges = vocab_size - len(self.vocab)
        if verbose:
            print(f"Running {num_merges} BPE merges (target vocab: {vocab_size})...")

        self.merges = []

        for i in range(num_merges):
            # Find most common adjacent pair
            pairs = self._get_pairs(words)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]

            if best_count < 2:
                break  # no more useful merges

            # Create new merged token
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.merges.append(best_pair)

            # Apply merge to all words
            words = self._apply_merge(words, best_pair, new_token)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i+1}/{num_merges} | vocab: {len(self.vocab)} | '{best_pair[0]}'+'{best_pair[1]}' → '{new_token}' (x{best_count})")

        # Build reverse vocab
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        self.trained = True

        if verbose:
            print(f"\nTokenizer trained!")
            print(f"  Final vocab size: {len(self.vocab)}")
            print(f"  Merges learned:   {len(self.merges)}")
            print(f"  Coverage: {self._coverage(text):.1f}% of text uses known tokens")

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Convert text → list of token IDs.

        Args:
            text:        input string
            add_special: wrap with BOS/EOS tokens
        """
        if not self.trained:
            raise RuntimeError("Tokenizer not trained. Call .train() first.")

        tokens = []

        if add_special:
            tokens.append(self.vocab[self.BOS])

        # Apply BPE to each word
        for word in self._split_to_words(text):
            word_tokens = list(word)

            # Apply merges in order
            for pair in self.merges:
                word_tokens = self._apply_merge_to_list(word_tokens, pair)

            # Convert to IDs
            for tok in word_tokens:
                if tok in self.vocab:
                    tokens.append(self.vocab[tok])
                else:
                    tokens.append(self.vocab[self.UNK])

        if add_special:
            tokens.append(self.vocab[self.EOS])

        return tokens

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Convert list of token IDs → text.

        Args:
            ids:          list of integer token IDs
            skip_special: don't include PAD/BOS/EOS in output
        """
        if not self.trained:
            raise RuntimeError("Tokenizer not trained.")

        tokens = []
        for id in ids:
            if id in self.vocab_r:
                tok = self.vocab_r[id]
                if skip_special and tok in self.SPECIAL_TOKENS:
                    continue
                tokens.append(tok)
            else:
                tokens.append(self.UNK)

        return "".join(tokens)

    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        data = {
            "vocab":   self.vocab,
            "merges":  [list(m) for m in self.merges],
            "trained": self.trained,
            "version": "mk1-tokenizer-v1",
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved: {path}")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab   = data["vocab"]
        tok.merges  = [tuple(m) for m in data["merges"]]
        tok.trained = data["trained"]
        tok.vocab_r = {v: k for k, v in tok.vocab.items()}
        print(f"Tokenizer loaded: {path} ({len(tok.vocab)} tokens)")
        return tok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_to_words(self, text: str) -> list[str]:
        """Split text into words, preserving spaces as part of words."""
        # GPT-style: space is attached to the FOLLOWING word
        return re.findall(r"\s?\S+", text)

    def _get_word_freqs(self, text: str) -> dict:
        """Build frequency dict of character-split words."""
        words = {}
        for word in self._split_to_words(text):
            chars = tuple(word)
            words[chars] = words.get(chars, 0) + 1
        return words

    def _get_pairs(self, words: dict) -> Counter:
        """Count all adjacent pairs across all words."""
        pairs = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _apply_merge(self, words: dict, pair: tuple, new_token: str) -> dict:
        """Apply a merge rule to all words in the corpus."""
        new_words = {}
        for word, freq in words.items():
            new_word = self._apply_merge_to_list(list(word), pair)
            new_words[tuple(new_word)] = new_words.get(tuple(new_word), 0) + freq
        return new_words

    def _apply_merge_to_list(self, tokens: list, pair: tuple) -> list:
        """Apply a single merge rule to a token list."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _coverage(self, text: str) -> float:
        """What % of encoded tokens are known (not UNK)."""
        ids = self.encode(text[:1000])
        unk_id = self.vocab[self.UNK]
        known = sum(1 for i in ids if i != unk_id)
        return 100 * known / len(ids) if ids else 0


# ------------------------------------------------------------------
# Download Shakespeare (our first training data)
# ------------------------------------------------------------------

def download_shakespeare(save_path: str = "data/shakespeare.txt"):
    """Download Shakespeare's complete works — public domain."""
    import urllib.request

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Shakespeare already downloaded: {save_path}")
        return save_path

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Shakespeare... ({url})")

    try:
        urllib.request.urlretrieve(url, save_path)
        size = os.path.getsize(save_path)
        print(f"Downloaded: {save_path} ({size:,} bytes)")
        return save_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating sample text instead...")

        # Fallback sample if no internet
        sample = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream.""" * 100

        with open(save_path, "w") as f:
            f.write(sample)
        return save_path


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  MK1 Tokenizer — Test Run")
    print("=" * 50)

    # Download Shakespeare
    path = download_shakespeare()

    # Train tokenizer
    tok = Tokenizer()
    tok.train(path, vocab_size=1000, verbose=True)

    # Save it
    tok.save("mk1/tokenizer.json")

    # Test encode/decode
    print("\n--- Encode/Decode Test ---")
    test = "To be, or not to be, that is the question"
    ids = tok.encode(test)
    decoded = tok.decode(ids)

    print(f"Original:  {test}")
    print(f"Token IDs: {ids[:20]}... ({len(ids)} tokens)")
    print(f"Decoded:   {decoded}")
    print(f"Match:     {test == decoded}")

    print(f"\nVocab size: {tok.vocab_size()}")
    print(f"\nTokenizer ready for MK1 model.")
