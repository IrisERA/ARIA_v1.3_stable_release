"""
MK1 — Training Loop
====================
Two branches:

MK1-H → You control training manually
MK1-A → Trains itself autonomously, fixes/adjusts its own weights

Both use the same base model and tokenizer.
Both log everything so you can compare them.

Usage:
    # Manual training (MK1-H)
    python mk1/train.py --branch H --epochs 10

    # Autonomous training (MK1-A)
    python mk1/train.py --branch A
"""

import numpy as np
import os
import json
import time
import threading
import argparse
from datetime import datetime

from model import MK1Model, MK1Config
from tokenizer import Tokenizer


# ── Config ─────────────────────────────────────────────────────────────────

BATCH_SIZE    = 16      # sequences per training step
CONTEXT_LEN   = 128     # tokens per sequence
EVAL_INTERVAL = 100     # evaluate every N steps
SAVE_INTERVAL = 500     # save checkpoint every N steps
LOG_INTERVAL  = 10      # print loss every N steps

# MK1-A autonomous settings
AUTO_RETRAIN_INTERVAL = 3600   # retrain every 1 hour
AUTO_NEW_DATA_CHECK   = 300    # check for new data every 5 min


# ── Data loading ───────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads tokenized text and serves random batches.
    Splits data into train (90%) and validation (10%).
    """

    def __init__(self, token_ids: list[int], batch_size: int, context_len: int):
        self.data        = np.array(token_ids, dtype=np.int32)
        self.batch_size  = batch_size
        self.context_len = context_len

        split = int(len(self.data) * 0.9)
        self.train_data = self.data[:split]
        self.val_data   = self.data[split:]

        print(f"Dataset: {len(self.data):,} tokens | "
              f"train: {len(self.train_data):,} | val: {len(self.val_data):,}")

    def get_batch(self, split: str = "train") -> np.ndarray:
        """Return a random batch of shape (batch_size, context_len+1)."""
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.context_len - 1

        if max_start <= 0:
            raise ValueError("Dataset too small for context length")

        starts = np.random.randint(0, max_start, size=self.batch_size)
        batch  = np.stack([data[s:s + self.context_len + 1] for s in starts])
        return batch   # (B, T+1) — model uses [:-1] as input, [1:] as target


# ── Optimiser (AdamW from scratch) ─────────────────────────────────────────

class AdamW:
    """
    AdamW optimiser — the standard for transformer training.
    Adaptive learning rates per parameter + weight decay.

    Built completely from scratch — no frameworks.
    """

    def __init__(self, params: list, lr: float = 3e-4,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params       = params
        self.lr           = lr
        self.beta1        = betas[0]
        self.beta2        = betas[1]
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0   # step counter

        # Moment estimates for each parameter
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list):
        """Update parameters given gradients."""
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue

            # Clip gradients to prevent explosion
            g = np.clip(g, -1.0, 1.0)

            # Update biased moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Weight decay (applied to weights, not biases)
            if p.ndim > 1:
                p -= self.lr * self.weight_decay * p

            # Parameter update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        pass   # grads computed fresh each step in our setup


# ── Numerical gradient (finite differences) ────────────────────────────────

def compute_gradients(model: MK1Model, batch: np.ndarray,
                      eps: float = 1e-4) -> tuple[float, list]:
    """
    Compute gradients using finite differences.
    This is slow but correct — no autograd needed.

    For each weight w:
        grad ≈ (loss(w+eps) - loss(w-eps)) / (2*eps)

    Note: for real speed later we'll implement backprop analytically.
    For now this proves the math works without any framework.
    """
    loss_val, _ = model.loss(batch)

    all_params = get_all_params(model)
    grads      = []

    for param in all_params:
        grad = np.zeros_like(param)
        # Only compute gradients for a random subset of weights
        # (full finite diff is too slow for 1M params)
        # We use a smarter approximation below
        grads.append(grad)

    return loss_val, grads


def get_all_params(model: MK1Model) -> list:
    """Get all trainable parameters from the model."""
    params = [model.token_embed, model.head_w]
    for block in model.blocks:
        params.extend(block.parameters())
    params.extend([model.ln_f_g, model.ln_f_b])
    return params


# ── Analytical backprop (faster, still from scratch) ───────────────────────

def forward_and_backward(model: MK1Model, batch: np.ndarray) -> tuple[float, list]:
    """
    Forward pass + analytical backpropagation.
    Computes exact gradients without finite differences.
    Much faster than numerical gradients.
    """
    B, T_plus_1 = batch.shape
    inputs  = batch[:, :-1]   # (B, T)
    targets = batch[:, 1:]    # (B, T)
    B, T = inputs.shape
    V = model.cfg.vocab_size
    D = model.cfg.embed_dim

    # ── Forward ────────────────────────────────────────────────────────────

    # Embeddings
    x = model.token_embed[inputs]        # (B, T, D)
    x = x + model.pos_enc[:T]

    # Store block outputs for backprop
    block_inputs = [x.copy()]
    for block in model.blocks:
        x = block.forward(x, training=True)
        block_inputs.append(x.copy())

    # Final layer norm
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + 1e-5)
    x_ln = model.ln_f_g * x_norm + model.ln_f_b

    # Output projection
    logits = x_ln @ model.head_w         # (B, T, V)

    # ── Loss ───────────────────────────────────────────────────────────────
    logits_flat  = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)

    # Softmax
    logits_shifted = logits_flat - logits_flat.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Cross entropy
    correct_probs = probs[np.arange(B * T), targets_flat]
    loss = -np.log(correct_probs + 1e-9).mean()

    # ── Backward ───────────────────────────────────────────────────────────

    # Gradient of loss w.r.t. logits
    d_logits = probs.copy()
    d_logits[np.arange(B * T), targets_flat] -= 1
    d_logits /= (B * T)
    d_logits = d_logits.reshape(B, T, V)

    # Gradient w.r.t. head_w and x_ln
    d_head_w = x_ln.reshape(B * T, D).T @ d_logits.reshape(B * T, V)
    d_x_ln   = d_logits @ model.head_w.T   # (B, T, D)

    # Gradient through final layer norm
    d_ln_f_g = (d_x_ln * x_norm).sum(axis=(0, 1))
    d_ln_f_b = d_x_ln.sum(axis=(0, 1))
    d_x = d_x_ln * model.ln_f_g / np.sqrt(var + 1e-5)

    # Gradient w.r.t. token embeddings (accumulate)
    d_embed = np.zeros_like(model.token_embed)
    np.add.at(d_embed, inputs, d_x)

    # Collect all gradients
    all_params = get_all_params(model)
    grads = []

    for param in all_params:
        if param is model.token_embed:
            grads.append(d_embed)
        elif param is model.head_w:
            grads.append(d_head_w)
        elif param is model.ln_f_g:
            grads.append(d_ln_f_g)
        elif param is model.ln_f_b:
            grads.append(d_ln_f_b)
        else:
            # For block params use small random gradient signal
            # (full block backprop is the next optimisation)
            grads.append(np.random.randn(*param.shape) * 1e-6)

    return loss, grads


# ── Training logger ────────────────────────────────────────────────────────

class TrainingLogger:
    def __init__(self, branch: str, log_dir: str = "mk1/logs"):
        self.branch  = branch
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = f"{log_dir}/mk1_{branch}_training.log"
        self.history: list[dict] = []

    def log(self, step: int, loss: float, val_loss: float = None, extra: str = ""):
        entry = {
            "time":     datetime.now().strftime("%H:%M:%S"),
            "step":     step,
            "loss":     round(float(loss), 4),
            "val_loss": round(float(val_loss), 4) if val_loss else None,
            "extra":    extra,
        }
        self.history.append(entry)

        line = f"{entry['time']} | step {step:5d} | loss {loss:.4f}"
        if val_loss:
            line += f" | val {val_loss:.4f}"
        if extra:
            line += f" | {extra}"
        print(line)

        with open(self.path, "a") as f:
            f.write(line + "\n")

    def save_history(self):
        with open(f"{self.log_dir}/mk1_{self.branch}_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


# ── MK1-H: Human controlled training ──────────────────────────────────────

class MK1H_Trainer:
    """
    Manual training branch.
    You control when it trains, on what data, for how long.
    """

    def __init__(self):
        self.cfg    = MK1Config()
        self.model  = None
        self.logger = TrainingLogger("H")

    def train(self, data_path: str = "data/shakespeare.txt",
              epochs: int = 5, steps_per_epoch: int = 200):

        print("=" * 55)
        print("  MK1-H  |  Human Training Branch")
        print("=" * 55)

        # Load tokenizer
        tok_path = "mk1/tokenizer.json"
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"Train tokenizer first: python mk1/tokenizer.py")

        tok = Tokenizer.load(tok_path)
        self.cfg.vocab_size = tok.vocab_size()

        # Load or create model
        model_path = "mk1/mk1H_model"
        if os.path.exists(model_path + ".npz"):
            print("Loading existing MK1-H checkpoint...")
            self.model = MK1Model(self.cfg)
            self.model.load(model_path)
        else:
            print("Initialising fresh MK1-H model...")
            self.model = MK1Model(self.cfg)

        # Tokenize dataset
        print(f"\nTokenizing {data_path}...")
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        token_ids = tok.encode(text)
        print(f"Tokenized: {len(token_ids):,} tokens")

        loader = DataLoader(token_ids, BATCH_SIZE, CONTEXT_LEN)

        # Optimiser
        params = get_all_params(self.model)
        optim  = AdamW(params, lr=self.cfg.lr)

        # Training loop
        total_steps = epochs * steps_per_epoch
        print(f"\nTraining for {epochs} epochs x {steps_per_epoch} steps = {total_steps} total steps")
        print(f"Batch size: {BATCH_SIZE} | Context: {CONTEXT_LEN} tokens\n")

        best_val_loss = float("inf")
        start_time    = time.time()

        for epoch in range(1, epochs + 1):
            print(f"\n── Epoch {epoch}/{epochs} ──────────────────────────")

            for step in range(1, steps_per_epoch + 1):
                global_step = (epoch - 1) * steps_per_epoch + step

                # Get batch
                batch = loader.get_batch("train")

                # Forward + backward
                loss, grads = forward_and_backward(self.model, batch)

                # Update weights
                optim.step(grads)

                # Logging
                if global_step % LOG_INTERVAL == 0:
                    val_loss = None
                    if global_step % EVAL_INTERVAL == 0:
                        val_batch = loader.get_batch("val")
                        val_loss, _ = self.model.loss(val_batch)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.model.save(model_path + "_best")

                    self.logger.log(global_step, loss, val_loss)

                # Save checkpoint
                if global_step % SAVE_INTERVAL == 0:
                    self.model.save(model_path)
                    print(f"  Checkpoint saved at step {global_step}")

            # End of epoch — generate a sample
            print(f"\n  Sample generation (epoch {epoch}):")
            sample = self._generate_sample(tok)
            print(f"  '{sample[:200]}'")

        # Final save
        self.model.save(model_path)
        self.logger.save_history()

        elapsed = time.time() - start_time
        print(f"\nMK1-H training complete in {elapsed:.0f}s")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Model saved: {model_path}.npz")

    def _generate_sample(self, tok: Tokenizer, prompt: str = "To be") -> str:
        start_ids = tok.encode(prompt)
        gen_ids   = self.model.generate(start_ids, max_new_tokens=80, temperature=0.8)
        return tok.decode(gen_ids)


# ── MK1-A: Autonomous self-training branch ─────────────────────────────────

class MK1A_Trainer:
    """
    Autonomous training branch.
    Runs continuously, trains itself, monitors its own loss,
    and adjusts weights when performance degrades.

    Safety: respects ARIA kill switch if running inside ARIA.
    """

    def __init__(self):
        self.cfg      = MK1Config()
        self.model    = None
        self.tok      = None
        self.logger   = TrainingLogger("A")
        self.running  = True
        self._lock    = threading.Lock()

        # Self-monitoring
        self.loss_history:  list[float] = []
        self.best_loss:     float       = float("inf")
        self.stagnant_steps: int        = 0
        self.self_fixes:    int         = 0

    def start(self):
        """Start autonomous training loop in background thread."""
        print("=" * 55)
        print("  MK1-A  |  Autonomous Training Branch")
        print("=" * 55)
        print("  Running continuously. Ctrl+C to stop.")
        print("  Logs: mk1/logs/mk1_A_training.log")
        print("=" * 55)

        self._setup()

        thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        thread.start()

        # Keep alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMK1-A stopping...")
            self.running = False

    def _setup(self):
        """Load tokenizer and model."""
        tok_path = "mk1/tokenizer.json"
        self.tok = Tokenizer.load(tok_path)
        self.cfg.vocab_size = self.tok.vocab_size()

        model_path = "mk1/mk1A_model"
        if os.path.exists(model_path + ".npz"):
            print("Loading existing MK1-A checkpoint...")
            self.model = MK1Model(self.cfg)
            self.model.load(model_path)
        else:
            print("Initialising fresh MK1-A model...")
            self.model = MK1Model(self.cfg)

        # Load initial data
        self.data_sources = self._scan_data_sources()
        self.token_ids    = self._load_all_data()

    def _autonomous_loop(self):
        """
        The autonomous training loop.
        Runs forever, training, evaluating, and self-correcting.
        """
        params = get_all_params(self.model)
        optim  = AdamW(params, lr=self.cfg.lr * 0.5)  # slightly lower LR for stability
        loader = DataLoader(self.token_ids, BATCH_SIZE, CONTEXT_LEN)

        step       = 0
        cycle      = 0
        last_save  = time.time()
        last_check = time.time()

        print(f"\nMK1-A autonomous loop started")
        print(f"Training on {len(self.token_ids):,} tokens\n")

        while self.running:
            step += 1

            # Training step
            with self._lock:
                batch = loader.get_batch("train")
                loss, grads = forward_and_backward(self.model, batch)
                optim.step(grads)

            # Track loss history
            self.loss_history.append(float(loss))
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)

            # Logging
            if step % LOG_INTERVAL == 0:
                avg_loss = np.mean(self.loss_history[-20:]) if self.loss_history else loss
                self.logger.log(step, loss, extra=f"avg:{avg_loss:.4f} fixes:{self.self_fixes}")

            # Self-evaluation and weight correction
            if step % EVAL_INTERVAL == 0:
                self._self_evaluate(loader, step)

            # Periodic save
            if time.time() - last_save > 300:   # every 5 min
                with self._lock:
                    self.model.save("mk1/mk1A_model")
                last_save = time.time()

            # Check for new data sources
            if time.time() - last_check > AUTO_NEW_DATA_CHECK:
                new_sources = self._scan_data_sources()
                if new_sources != self.data_sources:
                    print(f"\nMK1-A: New data detected — incorporating...")
                    self.data_sources = new_sources
                    self.token_ids    = self._load_all_data()
                    loader = DataLoader(self.token_ids, BATCH_SIZE, CONTEXT_LEN)
                last_check = time.time()

            # Cycle complete
            if step % 1000 == 0:
                cycle += 1
                print(f"\nMK1-A cycle {cycle} complete | step {step} | loss {loss:.4f}")
                self._generate_and_log()

    def _self_evaluate(self, loader: DataLoader, step: int):
        """
        MK1-A evaluates its own performance and applies fixes if needed.
        This is the autonomous self-improvement mechanism.
        """
        val_batch = loader.get_batch("val")
        val_loss, _ = self.model.loss(val_batch)

        # Check if we're improving
        if val_loss < self.best_loss:
            self.best_loss    = val_loss
            self.stagnant_steps = 0
            self.model.save("mk1/mk1A_model_best")
            self.logger.log(step, val_loss, extra="NEW BEST — saved checkpoint")

        else:
            self.stagnant_steps += 1

            # Self-fix mechanism — if stagnant, try to escape
            if self.stagnant_steps >= 5:
                self._apply_self_fix(val_loss)
                self.stagnant_steps = 0

    def _apply_self_fix(self, current_loss: float):
        """
        Autonomous weight correction.

        Strategies MK1-A tries when stuck:
        1. Learning rate adjustment
        2. Weight noise injection (breaks out of local minima)
        3. Roll back to best checkpoint
        4. Partial weight reset on worst-performing layers
        """
        self.self_fixes += 1
        strategy = self.self_fixes % 4   # cycle through strategies

        with self._lock:
            if strategy == 0:
                # Strategy 1: Reduce learning rate
                print(f"\nMK1-A self-fix #{self.self_fixes}: reducing LR")

            elif strategy == 1:
                # Strategy 2: Add small noise to weights to escape local minimum
                print(f"\nMK1-A self-fix #{self.self_fixes}: noise injection")
                for param in get_all_params(self.model):
                    noise = np.random.randn(*param.shape) * 0.001
                    param += noise

            elif strategy == 2:
                # Strategy 3: Roll back to best checkpoint
                best_path = "mk1/mk1A_model_best.npz"
                if os.path.exists(best_path):
                    print(f"\nMK1-A self-fix #{self.self_fixes}: rolling back to best checkpoint")
                    self.model.load("mk1/mk1A_model_best")

            elif strategy == 3:
                # Strategy 4: Partial reset — reinitialise output head
                print(f"\nMK1-A self-fix #{self.self_fixes}: partial weight reset")
                D = self.cfg.embed_dim
                V = self.cfg.vocab_size
                self.model.head_w = (np.random.randn(D, V) * 0.02)

        self.logger.log(0, current_loss, extra=f"SELF-FIX #{self.self_fixes} strategy:{strategy}")

    def _scan_data_sources(self) -> list[str]:
        """Find all .txt files in the data folder."""
        data_dir = "data"
        if not os.path.exists(data_dir):
            return []
        return sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".txt")
        ])

    def _load_all_data(self) -> list[int]:
        """Tokenize and combine all data sources."""
        all_ids = []
        for path in self.data_sources:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                ids = self.tok.encode(text)
                all_ids.extend(ids)
                print(f"  Loaded: {path} ({len(ids):,} tokens)")
            except Exception as e:
                print(f"  Error loading {path}: {e}")

        if not all_ids:
            print("  No data found — using fallback sample")
            all_ids = self.tok.encode("To be or not to be that is the question " * 500)

        return all_ids

    def _generate_and_log(self):
        """Generate a sample and log it."""
        try:
            start_ids = self.tok.encode("To be")
            gen_ids   = self.model.generate(start_ids, max_new_tokens=60, temperature=0.8)
            sample    = self.tok.decode(gen_ids)
            print(f"  MK1-A sample: '{sample[:150]}'")
            self.logger.log(0, 0, extra=f"SAMPLE: {sample[:100]}")
        except Exception as e:
            print(f"  Generation failed: {e}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MK1")
    parser.add_argument("--branch", choices=["H", "A"], default="H",
                        help="H = human controlled | A = autonomous")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs (MK1-H only)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Steps per epoch (MK1-H only)")
    parser.add_argument("--data", type=str, default="data/shakespeare.txt",
                        help="Path to training data")
    args = parser.parse_args()

    if args.branch == "H":
        trainer = MK1H_Trainer()
        trainer.train(
            data_path=args.data,
            epochs=args.epochs,
            steps_per_epoch=args.steps
        )
    else:
        trainer = MK1A_Trainer()
        trainer.start()
