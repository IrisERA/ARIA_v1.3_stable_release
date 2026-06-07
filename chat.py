"""
MK1 — Chat Interface
=====================
Clean chat UI for talking to MK1-H and MK1-A.
Claude-inspired design, dark theme, message bubbles.

Run with Python 3.11:
    py -3.11 mk1/chat.py

Requires:
    py -3.11 -m pip install PyQt6
"""

import sys
import os
import json
import threading
import time
from datetime import datetime

import torch
import torch_directml
import torch.nn.functional as F

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFrame, QScrollArea,
    QSizePolicy, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon, QPixmap, QPainter, QLinearGradient

# ── Device ─────────────────────────────────────────────────────────────────
import torch_directml

try:
    dml = torch_directml.device()
except:
    dml = torch_directml.device("cpu")

# ── Colours ────────────────────────────────────────────────────────────────
BG        = "#0f0f13"
SIDEBAR   = "#171720"
PANEL     = "#1c1c28"
PANEL2    = "#23232f"
BORDER    = "#2a2a3a"
USER_BG   = "#1e2d45"
MK1H_BG   = "#0f1f12"
MK1A_BG   = "#1a0f1f"
ACCENT_H  = "#00bfff"
ACCENT_A  = "#00ff88"
TEXT      = "#c8d8e8"
TEXT_HI   = "#f0f4f8"
TEXT_DIM  = "#4a5a6a"
MONO      = "Consolas"
SANS      = "Segoe UI"


# ── Model inference ────────────────────────────────────────────────────────

class MK1Inference:
    """Loads MK1 model and generates responses."""

    def __init__(self, branch: str):
        self.branch   = branch
        self.model    = None
        self.vocab    = None
        self.vocab_r  = None
        self.merges   = None
        self.loaded   = False
        self._load()

    def _load(self):
        try:
            # Load tokenizer
            tok_path = "mk1/tokenizer.json"
            with open(tok_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocab   = data["vocab"]
            self.vocab_r = {v: k for k, v in self.vocab.items()}
            self.merges  = [tuple(m) for m in data["merges"]]

            # Load model
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from train_gpu import MK1GPU, MK1Config

            cfg = MK1Config()
            cfg.vocab_size = len(self.vocab)
            self.model = MK1GPU(cfg).to("cpu")

            model_path = f"mk1/mk1{self.branch}_gpu_model_best.pt"
            if not os.path.exists(model_path):
                model_path = f"mk1/mk1{self.branch}_gpu_model.pt"
            if not os.path.exists(model_path):
                model_path = "mk1/mk1_BASE.pt"

            if os.path.exists(model_path):
                self.model.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=False)
                )
                self.loaded = True
                print(f"MK1-{self.branch} loaded from {model_path}")
            else:
                print(f"No model found for MK1-{self.branch}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Load error: {e}")

    def encode(self, text: str) -> list:
        import re
        unk = self.vocab.get("<UNK>", 0)
        ids = []
        for word in re.findall(r"\s?\S+", text):
            toks = list(word)
            for pair in self.merges:
                i, new = 0, []
                while i < len(toks):
                    if i < len(toks)-1 and toks[i] == pair[0] and toks[i+1] == pair[1]:
                        new.append(pair[0]+pair[1]); i += 2
                    else:
                        new.append(toks[i]); i += 1
                toks = new
            ids.extend(self.vocab.get(t, unk) for t in toks)
        return ids

    def decode(self, ids: list) -> str:
        special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        return "".join(self.vocab_r.get(i, "?") for i in ids
                      if self.vocab_r.get(i, "?") not in special)

    def generate(self, prompt: str, max_new: int = 150,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        if not self.loaded or self.model is None:
            return "MK1 model not loaded. Train first."

        self.model.eval()
        start_ids = self.encode(prompt)
        if not start_ids:
            start_ids = [4]

        ids = torch.tensor([start_ids], dtype=torch.long, device="cpu")

        with torch.no_grad():
            for _ in range(max_new):
                ctx    = ids[:, -self.model.cfg.context_len:]
                logits = self.model(ctx)[:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, -1:]] = -float('inf')
                probs   = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                ids     = torch.cat([ids, next_id], dim=1)

                # Stop at newline (end of reply turn)
                token = self.vocab_r.get(int(next_id[0]), "")
                if token == "\n" and len(ids[0]) > len(start_ids) + 10:
                    break
                # Also stop if we hit "User:" — model started a new turn
                decoded_so_far = self.decode(ids[0][len(start_ids):].tolist())
                if "User:" in decoded_so_far or f"MK1-" in decoded_so_far[10:]:
                    break

        generated = ids[0][len(start_ids):].tolist()
        result = self.decode(generated).strip()

        # Clean up any leaked turn markers
        for marker in [f"User:", f"MK1-H:", f"MK1-A:"]:
            if marker in result:
                result = result[:result.index(marker)].strip()

        return result if result else "..."


# ── Inference worker ───────────────────────────────────────────────────────

class InferenceWorker(QThread):
    result_signal = pyqtSignal(str)
    error_signal  = pyqtSignal(str)

    def __init__(self, inference: MK1Inference, prompt: str):
        super().__init__()
        self.inference = inference
        self.prompt    = prompt

    def run(self):
        try:
            response = self.inference.generate(self.prompt)
            self.result_signal.emit(response)
        except Exception as e:
            self.error_signal.emit(str(e))


# ── Message bubble ─────────────────────────────────────────────────────────

class MessageBubble(QWidget):
    def __init__(self, text: str, role: str, branch: str = "H"):
        super().__init__()
        self.role   = role
        self.branch = branch
        self._build(text)

    def _build(self, text: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)

        is_user = self.role == "user"
        accent  = ACCENT_H if self.branch == "H" else ACCENT_A
        bg      = USER_BG if is_user else (MK1H_BG if self.branch == "H" else MK1A_BG)
        border  = "#2a4060" if is_user else (ACCENT_H + "30" if self.branch == "H" else ACCENT_A + "30")

        if is_user:
            layout.addStretch()

        # Avatar
        if not is_user:
            avatar = QLabel(f"{'H' if self.branch == 'H' else 'A'}")
            avatar.setFixedSize(28, 28)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            avatar.setFont(QFont(MONO, 10, QFont.Weight.Bold))
            avatar.setStyleSheet(f"""
                background: {accent}20;
                color: {accent};
                border: 1px solid {accent}60;
                border-radius: 14px;
            """)
            layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)

        # Bubble
        bubble = QWidget()
        bubble.setStyleSheet(f"""
            QWidget {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
        """)
        bl = QVBoxLayout(bubble)
        bl.setContentsMargins(14, 10, 14, 10)

        # Role label
        if not is_user:
            role_lbl = QLabel(f"MK1-{self.branch}")
            role_lbl.setFont(QFont(MONO, 8))
            role_lbl.setStyleSheet(f"color: {accent}; background: transparent; border: none;")
            bl.addWidget(role_lbl)

        # Text
        msg = QLabel(text)
        msg.setFont(QFont(SANS, 11))
        msg.setStyleSheet(f"color: {TEXT_HI}; background: transparent; border: none;")
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        bl.addWidget(msg)

        # Timestamp
        ts = QLabel(datetime.now().strftime("%H:%M"))
        ts.setFont(QFont(MONO, 8))
        ts.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
        ts.setAlignment(Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft)
        bl.addWidget(ts)

        layout.addWidget(bubble)

        if is_user:
            # User avatar
            avatar = QLabel("U")
            avatar.setFixedSize(28, 28)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            avatar.setFont(QFont(MONO, 10, QFont.Weight.Bold))
            avatar.setStyleSheet(f"""
                background: #2a3a4a;
                color: {TEXT};
                border: 1px solid #3a4a5a;
                border-radius: 14px;
            """)
            layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        else:
            layout.addStretch()


class TypingIndicator(QWidget):
    def __init__(self, branch: str):
        super().__init__()
        self.branch = branch
        self._dot   = 0
        self._build()
        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(400)

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        accent = ACCENT_H if self.branch == "H" else ACCENT_A

        avatar = QLabel(self.branch)
        avatar.setFixedSize(28, 28)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setFont(QFont(MONO, 10, QFont.Weight.Bold))
        avatar.setStyleSheet(f"""
            background: {accent}20; color: {accent};
            border: 1px solid {accent}60; border-radius: 14px;
        """)
        layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)

        self.dots = QLabel("MK1 is thinking...")
        self.dots.setFont(QFont(MONO, 10))
        self.dots.setStyleSheet(f"color: {accent}; background: transparent;")
        layout.addWidget(self.dots)
        layout.addStretch()

    def _animate(self):
        dots = ["MK1 is thinking   ", "MK1 is thinking.  ", "MK1 is thinking.. ", "MK1 is thinking..."]
        self.dots.setText(dots[self._dot % 4])
        self._dot += 1


# ── Chat panel ─────────────────────────────────────────────────────────────

class ChatPanel(QWidget):
    def __init__(self, branch: str):
        super().__init__()
        self.branch    = branch
        self.accent    = ACCENT_H if branch == "H" else ACCENT_A
        self.inference = None
        self.worker    = None
        self.history   = []   # list of {"role": "user"/"mk1", "content": str}
        self._build()
        self._load_model()

    def _load_model(self):
        def load():
            self.inference = MK1Inference(self.branch)
            if self.inference.loaded:
                self._add_message(
                    f"MK1-{self.branch} is online. Model loaded successfully.",
                    "system"
                )
            else:
                self._add_message(
                    f"MK1-{self.branch} model not found. Train first using the terminal.",
                    "system"
                )
        threading.Thread(target=load, daemon=True).start()

    def _build_prompt(self, user_text: str) -> str:
        """
        Build a conversational prompt from history.
        Format matches conversations.txt training data:
            User: <msg>
            MK1-H: <reply>
            ...
        """
        lines = []
        # Include last 6 messages (3 exchanges) for context
        for msg in self.history[-6:]:
            if msg["role"] == "user":
                lines.append(f"User: {msg['content']}")
            else:
                lines.append(f"MK1-{self.branch}: {msg['content']}")

        # Add current user message
        lines.append(f"User: {user_text}")
        # Prime the model to reply
        lines.append(f"MK1-{self.branch}:")

        return "\n".join(lines)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(f"background: {SIDEBAR}; border-bottom: 1px solid {BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)

        dot = QLabel("●")
        dot.setFont(QFont(MONO, 12))
        dot.setStyleSheet(f"color: {self.accent}; background: transparent;")
        hl.addWidget(dot)

        name = QLabel(f"MK1-{self.branch}")
        name.setFont(QFont(SANS, 13, QFont.Weight.Bold))
        name.setStyleSheet(f"color: {TEXT_HI}; background: transparent;")
        hl.addWidget(name)

        desc = QLabel("Human Branch" if self.branch == "H" else "Autonomous Branch")
        desc.setFont(QFont(MONO, 9))
        desc.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; margin-left: 8px;")
        hl.addWidget(desc)
        hl.addStretch()

        clear_btn = QPushButton("clear")
        clear_btn.setFont(QFont(MONO, 9))
        clear_btn.setFixedSize(60, 26)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 2px;
            }}
            QPushButton:hover {{ color: {self.accent}; border-color: {self.accent}; }}
        """)
        clear_btn.clicked.connect(self._clear)
        hl.addWidget(clear_btn)

        layout.addWidget(header)

        # Messages scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:vertical {{
                background: {PANEL}; width: 4px; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER}; border-radius: 2px;
            }}
        """)

        self.msg_container = QWidget()
        self.msg_container.setStyleSheet(f"background: {BG};")
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setContentsMargins(0, 16, 0, 16)
        self.msg_layout.setSpacing(8)
        self.msg_layout.addStretch()

        self.scroll.setWidget(self.msg_container)
        layout.addWidget(self.scroll, 1)

        # Input bar
        input_bar = QWidget()
        input_bar.setStyleSheet(f"background: {SIDEBAR}; border-top: 1px solid {BORDER};")
        il = QHBoxLayout(input_bar)
        il.setContentsMargins(16, 12, 16, 12)
        il.setSpacing(10)

        self.input = QLineEdit()
        self.input.setFont(QFont(SANS, 11))
        self.input.setPlaceholderText(f"Message MK1-{self.branch}...")
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {PANEL};
                color: {TEXT_HI};
                border: 1px solid {BORDER};
                border-radius: 20px;
                padding: 10px 18px;
            }}
            QLineEdit:focus {{
                border-color: {self.accent}60;
            }}
        """)
        self.input.returnPressed.connect(self._send)
        il.addWidget(self.input, 1)

        self.send_btn = QPushButton("▶")
        self.send_btn.setFont(QFont(MONO, 12))
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {self.accent};
                color: #000;
                border: none;
                border-radius: 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: white; }}
            QPushButton:disabled {{ background: {BORDER}; color: {TEXT_DIM}; }}
        """)
        self.send_btn.clicked.connect(self._send)
        il.addWidget(self.send_btn)

        layout.addWidget(input_bar)

    def _send(self):
        text = self.input.text().strip()
        if not text or self.worker:
            return

        self.input.clear()
        self.send_btn.setEnabled(False)

        # Add user message to UI and history
        self._add_message(text, "user")
        self.history.append({"role": "user", "content": text})

        # Show typing indicator
        self.typing = TypingIndicator(self.branch)
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, self.typing)
        self._scroll_to_bottom()

        # Build conversational prompt and generate
        if self.inference and self.inference.loaded:
            prompt = self._build_prompt(text)
            self.worker = InferenceWorker(self.inference, prompt)
            self.worker.result_signal.connect(self._on_response)
            self.worker.error_signal.connect(self._on_error)
            self.worker.start()
        else:
            self._remove_typing()
            self._add_message("Model not loaded yet. Please wait.", "system")
            self.send_btn.setEnabled(True)

    def _on_response(self, response: str):
        self._remove_typing()
        self._add_message(response, "mk1")
        self.history.append({"role": "mk1", "content": response})
        self.worker = None
        self.send_btn.setEnabled(True)

    def _on_error(self, error: str):
        self._remove_typing()
        self._add_message(f"Error: {error}", "system")
        self.worker = None
        self.send_btn.setEnabled(True)

    def _add_message(self, text: str, role: str):
        if role in ("user", "mk1"):
            bubble = MessageBubble(text, role, self.branch)
            self.msg_layout.insertWidget(self.msg_layout.count() - 1, bubble)
        else:
            # System message
            lbl = QLabel(text)
            lbl.setFont(QFont(MONO, 9))
            lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 4px 20px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            self.msg_layout.insertWidget(self.msg_layout.count() - 1, lbl)

        self._scroll_to_bottom()

    def _remove_typing(self):
        if hasattr(self, 'typing') and self.typing:
            self.typing.setParent(None)
            self.typing = None

    def _scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def _clear(self):
        while self.msg_layout.count() > 1:
            item = self.msg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.history.clear()


# ── Sidebar ────────────────────────────────────────────────────────────────

class Sidebar(QWidget):
    branch_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        self.setStyleSheet(f"background: {SIDEBAR}; border-right: 1px solid {BORDER};")
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo
        logo = QWidget()
        logo.setFixedHeight(60)
        logo.setStyleSheet(f"background: {SIDEBAR}; border-bottom: 1px solid {BORDER};")
        ll = QHBoxLayout(logo)
        ll.setContentsMargins(16, 0, 16, 0)

        icon = QLabel("⬡")
        icon.setFont(QFont(SANS, 18))
        icon.setStyleSheet(f"color: {ACCENT_H}; background: transparent;")
        ll.addWidget(icon)

        name = QLabel("MK1")
        name.setFont(QFont(SANS, 16, QFont.Weight.Bold))
        name.setStyleSheet(f"color: {TEXT_HI}; background: transparent;")
        ll.addWidget(name)
        ll.addStretch()

        layout.addWidget(logo)

        # Branch buttons
        layout.addSpacing(12)

        branches_label = QLabel("BRANCHES")
        branches_label.setFont(QFont(MONO, 8))
        branches_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 0 16px;")
        layout.addWidget(branches_label)

        layout.addSpacing(6)

        self.h_btn = self._branch_btn("MK1-H", "Human Branch", ACCENT_H, 0)
        self.a_btn = self._branch_btn("MK1-A", "Autonomous Branch", ACCENT_A, 1)
        layout.addWidget(self.h_btn)
        layout.addWidget(self.a_btn)

        layout.addSpacing(16)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        layout.addWidget(line)

        layout.addSpacing(12)

        # Both button
        both_label = QLabel("SIMULTANEOUS")
        both_label.setFont(QFont(MONO, 8))
        both_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 0 16px;")
        layout.addWidget(both_label)

        layout.addSpacing(6)

        both_btn = self._branch_btn("Both Branches", "H + A side by side", "#7c4dff", 2)
        layout.addWidget(both_btn)

        layout.addStretch()

        # Version
        ver = QLabel("MK1 v0.1.0")
        ver.setFont(QFont(MONO, 8))
        ver.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 12px 16px;")
        layout.addWidget(ver)

        # Set H active by default
        self.h_btn.setProperty("active", True)
        self.h_btn.style().unpolish(self.h_btn)
        self.h_btn.style().polish(self.h_btn)

    def _branch_btn(self, name: str, desc: str, color: str, index: int) -> QPushButton:
        btn = QPushButton()
        btn.setFixedHeight(52)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 6px;
                margin: 2px 8px;
                text-align: left;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background: {color}15;
            }}
        """)

        bl = QHBoxLayout(btn)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        dot = QLabel("●")
        dot.setFont(QFont(MONO, 10))
        dot.setStyleSheet(f"color: {color}; background: transparent;")
        dot.setFixedWidth(16)
        bl.addWidget(dot)

        text_col = QWidget()
        text_col.setStyleSheet("background: transparent;")
        tl = QVBoxLayout(text_col)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(1)

        n = QLabel(name)
        n.setFont(QFont(SANS, 11, QFont.Weight.Bold))
        n.setStyleSheet(f"color: {TEXT_HI}; background: transparent;")
        tl.addWidget(n)

        d = QLabel(desc)
        d.setFont(QFont(MONO, 8))
        d.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        tl.addWidget(d)

        bl.addWidget(text_col, 1)

        btn.clicked.connect(lambda: self.branch_changed.emit(index))
        return btn


# ── Main window ────────────────────────────────────────────────────────────

class MK1Chat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MK1 Chat")
        self.setMinimumSize(900, 640)
        self.resize(1100, 720)
        self._apply_palette()
        self._build()

    def _apply_palette(self):
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(BG))
        p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
        p.setColor(QPalette.ColorRole.Base, QColor(PANEL))
        p.setColor(QPalette.ColorRole.Text, QColor(TEXT))
        self.setPalette(p)
        self.setStyleSheet(f"background: {BG};")

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.branch_changed.connect(self._switch_branch)
        root.addWidget(self.sidebar)

        # Chat panels
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {BG};")

        self.h_panel = ChatPanel("H")
        self.a_panel = ChatPanel("A")

        # Both panels side by side
        both_widget = QWidget()
        both_layout = QHBoxLayout(both_widget)
        both_layout.setContentsMargins(0, 0, 0, 0)
        both_layout.setSpacing(1)
        both_layout.addWidget(ChatPanel("H"))
        both_layout.addWidget(ChatPanel("A"))

        self.stack.addWidget(self.h_panel)   # index 0
        self.stack.addWidget(self.a_panel)   # index 1
        self.stack.addWidget(both_widget)    # index 2

        root.addWidget(self.stack, 1)

    def _switch_branch(self, index: int):
        self.stack.setCurrentIndex(index)


# ── Entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    window = MK1Chat()
    window.show()
    sys.exit(app.exec())
