"""
MK1 — Dual Branch Terminal
===========================
Side by side terminals for MK1-H and MK1-A.
Watch both branches train simultaneously.
Control each independently.

NOTE:
One  = 50%  gpu usage
Both = 100% gpu usage

Run with Python 3.11:
    py -3.11 mk1/terminal.py

Requires:
    py -3.11 -m pip install PyQt6
"""

import sys
import os
import time
import threading
import subprocess
import json
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame, QSplitter, QProgressBar,
    QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QProcess
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon, QPixmap

# ── Colours ────────────────────────────────────────────────────────────────
BG       = "#050810"
PANEL    = "#0a0f1a"
PANEL2   = "#0f1628"
BORDER   = "#1a2640"
H_COLOR  = "#00bfff"   # MK1-H — blue
A_COLOR  = "#00ff88"   # MK1-A — green
RED      = "#ff1744"
YELLOW   = "#ffd600"
TEXT     = "#8ab8d4"
TEXT_HI  = "#e0f0ff"
DIM      = "#304060"
MONO     = "Consolas"
SANS     = "Segoe UI"


def styled_btn(text, color, width=100):
    btn = QPushButton(text)
    btn.setFont(QFont(MONO, 9))
    btn.setFixedHeight(28)
    if width:
        btn.setFixedWidth(width)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: transparent;
            color: {color};
            border: 1px solid {color};
            border-radius: 2px;
            letter-spacing: 1px;
            padding: 0 8px;
        }}
        QPushButton:hover {{
            background: {color};
            color: #000;
        }}
        QPushButton:disabled {{
            color: {DIM};
            border-color: {DIM};
        }}
    """)
    return btn


class TrainingWorker(QThread):
    """Runs training in background, emits log lines and stats."""
    log_signal    = pyqtSignal(str, str)   # line, level
    stats_signal  = pyqtSignal(float, float, int)  # loss, val_loss, step
    sample_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)  # running/stopped/paused

    def __init__(self, branch: str, epochs: int = 100, steps: int = 2000, lr: float = 0.0002, sample_text: str = "To be"):
        super().__init__()
        self.branch      = branch
        self.epochs      = epochs
        self.steps       = steps
        self.lr          = lr
        self.sample_text = sample_text
        self.process     = None
        self._stop       = False
        self._pause      = False

    def run(self):
        self.status_signal.emit("running")
        self.log_signal.emit(f"Starting MK1-{self.branch} training...", "info")

        cmd = [
            "py", "-3.11",
            os.path.join("mk1", "train_gpu.py"),
            "--branch", self.branch,
            "--epochs", str(self.epochs),
            "--steps", str(self.steps),
            "--lr", str(self.lr),
            "--sample", self.sample_text
        ]

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONUTF8"] = "1"
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env=env
            )

            for line in self.process.stdout:
                if self._stop:
                    break

                line = line.rstrip()
                if not line:
                    continue

                # Parse loss from log lines
                if "| loss" in line:
                    try:
                        parts = line.split("|")
                        step_part = [p for p in parts if "step" in p]
                        loss_part = [p for p in parts if "loss" in p and "val" not in p]
                        val_part  = [p for p in parts if "val" in p]

                        step = int(step_part[0].split()[-1]) if step_part else 0
                        loss = float(loss_part[0].split()[-1]) if loss_part else 0
                        val  = float(val_part[0].split()[-1]) if val_part else 0

                        self.stats_signal.emit(loss, val, step)
                        level = "ok" if loss < 2.5 else "warn" if loss < 3.5 else "info"
                        self.log_signal.emit(line, level)
                    except:
                        self.log_signal.emit(line, "info")

                elif "Sample:" in line:
                    self.log_signal.emit(line, "sample")
                    # Start collecting multi-line sample
                    self._collecting_sample = True
                    self._sample_lines = [line.replace("Sample:", "").strip().strip("'")]

                elif hasattr(self, "_collecting_sample") and self._collecting_sample:
                    # Keep collecting until we hit a non-sample line
                    if line.startswith("Saved:") or "-- Epoch" in line or "| loss" in line or line.startswith("Done"):
                        # End of sample
                        full_sample = " ".join(self._sample_lines).strip().strip("'")
                        self.sample_signal.emit(full_sample)
                        self._collecting_sample = False
                        self._sample_lines = []
                        # Still process this line normally
                        if "| loss" in line:
                            try:
                                parts = line.split("|")
                                step_part = [p for p in parts if "step" in p]
                                loss_part = [p for p in parts if "loss" in p and "val" not in p]
                                val_part  = [p for p in parts if "val" in p]
                                step = int(step_part[0].split()[-1]) if step_part else 0
                                loss = float(loss_part[0].split()[-1]) if loss_part else 0
                                val  = float(val_part[0].split()[-1]) if val_part else 0
                                self.stats_signal.emit(loss, val, step)
                            except:
                                pass
                        self.log_signal.emit(line, "info")
                    else:
                        self._sample_lines.append(line.strip().strip("'"))
                        self.log_signal.emit(line, "sample")

                elif "Saved:" in line or "saved" in line.lower():
                    self.log_signal.emit(line, "save")

                elif "error" in line.lower() or "Error" in line:
                    self.log_signal.emit(line, "error")

                else:
                    self.log_signal.emit(line, "info")

            self.process.wait()

        except Exception as e:
            self.log_signal.emit(f"Error: {e}", "error")

        self.status_signal.emit("stopped")
        self.log_signal.emit(f"MK1-{self.branch} training stopped.", "warn")

    def stop(self):
        self._stop = True
        if self.process:
            self.process.terminate()

    def pause(self):
        if self.process:
            self.process.send_signal(__import__('signal').SIGSTOP if hasattr(__import__('signal'), 'SIGSTOP') else 0)

    def resume(self):
        if self.process:
            self.process.send_signal(__import__('signal').SIGCONT if hasattr(__import__('signal'), 'SIGCONT') else 0)


class BranchPanel(QWidget):
    """A single branch terminal panel."""

    def __init__(self, branch: str, color: str):
        super().__init__()
        self.branch  = branch
        self.color   = color
        self.worker  = None
        self.running = False
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet(f"background: {PANEL}; border-bottom: 1px solid {BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 0, 12, 0)

        # Branch indicator
        dot = QLabel("●")
        dot.setFont(QFont(MONO, 14))
        dot.setStyleSheet(f"color: {self.color}; background: transparent;")
        hl.addWidget(dot)

        title = QLabel(f"MK1-{self.branch}")
        title.setFont(QFont(SANS, 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_HI}; background: transparent;")
        hl.addWidget(title)

        subtitle = QLabel("Human Branch" if self.branch == "H" else "Autonomous Branch")
        subtitle.setFont(QFont(MONO, 9))
        subtitle.setStyleSheet(f"color: {DIM}; background: transparent; margin-left: 6px;")
        hl.addWidget(subtitle)

        hl.addStretch()

        # Status badge
        self.status_badge = QLabel("● IDLE")
        self.status_badge.setFont(QFont(MONO, 9))
        self.status_badge.setStyleSheet(f"color: {DIM}; background: transparent;")
        hl.addWidget(self.status_badge)

        layout.addWidget(header)

        # Stats bar
        stats = QWidget()
        stats.setFixedHeight(36)
        stats.setStyleSheet(f"background: {PANEL2}; border-bottom: 1px solid {BORDER};")
        sl = QHBoxLayout(stats)
        sl.setContentsMargins(12, 0, 12, 0)
        sl.setSpacing(20)

        self.loss_label = self._stat_label("LOSS", "--")
        self.val_label  = self._stat_label("VAL", "--")
        self.step_label = self._stat_label("STEP", "0")
        self.eta_label  = self._stat_label("ETA", "--")
        sl.addWidget(self.loss_label[0])
        sl.addWidget(self.loss_label[1])
        sl.addWidget(self.val_label[0])
        sl.addWidget(self.val_label[1])
        sl.addWidget(self.step_label[0])
        sl.addWidget(self.step_label[1])
        sl.addWidget(self.eta_label[0])
        sl.addWidget(self.eta_label[1])
        sl.addStretch()

        # Loss progress bar
        self.loss_bar = QProgressBar()
        self.loss_bar.setRange(0, 300)
        self.loss_bar.setValue(0)
        self.loss_bar.setTextVisible(False)
        self.loss_bar.setFixedWidth(120)
        self.loss_bar.setFixedHeight(6)
        self.loss_bar.setStyleSheet(f"""
            QProgressBar {{ background: {BORDER}; border: none; border-radius: 2px; }}
            QProgressBar::chunk {{ background: {self.color}; border-radius: 2px; }}
        """)
        sl.addWidget(self.loss_bar)

        layout.addWidget(stats)

        # Terminal log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont(MONO, 10))
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: {BG};
                color: {TEXT};
                border: none;
                padding: 8px;
            }}
        """)
        layout.addWidget(self.log, 1)

        # Sample output box
        sample_header = QWidget()
        sample_header.setFixedHeight(28)
        sample_header.setStyleSheet(f"background: {PANEL2}; border-top: 1px solid {BORDER};")
        shl = QHBoxLayout(sample_header)
        shl.setContentsMargins(12, 0, 12, 0)
        shl.addWidget(QLabel("// LATEST SAMPLE"))
        layout.addWidget(sample_header)

        self.sample_box = QTextEdit()
        self.sample_box.setReadOnly(True)
        self.sample_box.setFont(QFont(MONO, 10))
        self.sample_box.setFixedHeight(80)
        self.sample_box.setStyleSheet(f"""
            QTextEdit {{
                background: {PANEL};
                color: {self.color};
                border: none;
                border-top: 1px solid {BORDER};
                padding: 8px;
            }}
        """)
        layout.addWidget(self.sample_box)

        # Controls
        controls = QWidget()
        controls.setFixedHeight(44)
        controls.setStyleSheet(f"background: {PANEL}; border-top: 1px solid {BORDER};")
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(12, 0, 12, 0)
        cl.setSpacing(8)

        self.start_btn = styled_btn("▶ START", self.color, 90)
        self.stop_btn  = styled_btn("■ STOP", RED, 90)
        self.stop_btn.setEnabled(False)

        # Config inputs
        input_style = f"""
            QLineEdit {{
                background: {PANEL2};
                color: {TEXT_HI};
                border: 1px solid {BORDER};
                border-radius: 2px;
                padding: 2px 6px;
            }}
        """
        cl.addWidget(QLabel("LR:"))
        self.lr_input = QLineEdit("0.0002")
        self.lr_input.setFont(QFont(MONO, 9))
        self.lr_input.setFixedWidth(70)
        self.lr_input.setFixedHeight(26)
        self.lr_input.setStyleSheet(input_style)

        cl.addWidget(QLabel("Epochs:"))
        self.epoch_input = QLineEdit("100")
        self.epoch_input.setFont(QFont(MONO, 9))
        self.epoch_input.setFixedWidth(50)
        self.epoch_input.setFixedHeight(26)
        self.epoch_input.setStyleSheet(input_style)

        cl.addWidget(QLabel("Sample:"))
        self.sample_input = QLineEdit("To be")
        self.sample_input.setFont(QFont(MONO, 9))
        self.sample_input.setFixedWidth(120)
        self.sample_input.setFixedHeight(26)
        self.sample_input.setPlaceholderText("sample prompt...")
        self.sample_input.setStyleSheet(input_style)
        self.sample_input.setToolTip("Text used to generate sample output during training")

        cl.addWidget(self.lr_input)
        cl.addWidget(self.epoch_input)
        cl.addWidget(self.sample_input)
        cl.addStretch()
        cl.addWidget(self.start_btn)
        cl.addWidget(self.stop_btn)

        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)

        layout.addWidget(controls)

        # Style labels
        for lbl in self.findChildren(QLabel):
            if lbl.text() in ("LR:", "Epochs:", "// LATEST SAMPLE"):
                lbl.setFont(QFont(MONO, 9))
                lbl.setStyleSheet(f"color: {DIM}; background: transparent;")

    def _stat_label(self, title, value):
        t = QLabel(title)
        t.setFont(QFont(MONO, 8))
        t.setStyleSheet(f"color: {DIM}; background: transparent;")
        v = QLabel(value)
        v.setFont(QFont(MONO, 11))
        v.setStyleSheet(f"color: {self.color}; background: transparent;")
        v.setObjectName(f"{title}_val")
        return t, v

    def start_training(self):
        lr     = float(self.lr_input.text())
        epochs = int(self.epoch_input.text())

        # Reset ETA tracking
        if hasattr(self, '_eta_start'):
            del self._eta_start
            del self._eta_step0

        sample_text = self.sample_input.text().strip() or "To be"
        self.worker = TrainingWorker(self.branch, epochs=epochs, steps=2000, lr=lr, sample_text=sample_text)
        self.worker.log_signal.connect(self._on_log)
        self.worker.stats_signal.connect(self._on_stats)
        self.worker.sample_signal.connect(self._on_sample)
        self.worker.status_signal.connect(self._on_status)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.running = True

    def stop_training(self):
        if self.worker:
            self.worker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.running = False

    def _on_log(self, line: str, level: str):
        colors = {
            "info":   TEXT,
            "ok":     "#00e676",
            "warn":   YELLOW,
            "error":  RED,
            "sample": self.color,
            "save":   "#7c4dff",
        }
        color = colors.get(level, TEXT)
        t = datetime.now().strftime("%H:%M:%S")
        self.log.append(
            f'<span style="color:{DIM};">{t}</span> '
            f'<span style="color:{color};">{line}</span>'
        )
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _loss_color(self, v):
        if v <= 1.5:  return "#00bfff"   # blue
        if v <= 2.0:  return "#006400"   # dark green
        if v <= 2.5:  return "#00e676"   # light green
        if v <= 3.0:  return "#ff9100"   # orange
        return "#ff1744"                 # red

    def _on_stats(self, loss: float, val: float, step: int):
        loss_color = self._loss_color(loss)
        val_color  = self._loss_color(val)
        self.loss_label[1].setText(f"{loss:.4f}")
        self.loss_label[1].setStyleSheet(f"color: {loss_color}; background: transparent;")

        if val > 0:
            self.val_label[1].setText(f"{val:.4f}")
            self.val_label[1].setStyleSheet(f"color: {val_color}; background: transparent;")
        self.step_label[1].setText(f"{step:,}")

        # Progress bar: 4.0 = start, 1.0 = target, full = 300
        progress = max(0, min(300, int((4.0 - loss) / 3.0 * 300)))
        self.loss_bar.setValue(progress)

        # ETA estimate based on steps/time
        now = time.time()
        if not hasattr(self, '_eta_start'):
            self._eta_start = now
            self._eta_step0 = step
        elif step > self._eta_step0:
            elapsed = now - self._eta_start
            steps_done = step - self._eta_step0
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            total_steps = int(self.epoch_input.text()) * 2000
            remaining = max(0, total_steps - step)
            if steps_per_sec > 0:
                secs = int(remaining / steps_per_sec)
                h, m = divmod(secs // 60, 60)
                eta = f"{h}h{m:02d}m" if h > 0 else f"{m}m{secs%60:02d}s"
                self.eta_label[1].setText(eta)
                self.eta_label[1].setStyleSheet(f"color: {DIM}; background: transparent;")

    def _on_sample(self, sample: str):
        self.sample_box.setPlainText(sample[:300])

    def _on_status(self, status: str):
        colors = {"running": self.color, "stopped": DIM, "paused": YELLOW}
        labels = {"running": "● RUNNING", "stopped": "● IDLE", "paused": "● PAUSED"}
        self.status_badge.setText(labels.get(status, "● IDLE"))
        self.status_badge.setStyleSheet(f"color: {colors.get(status, DIM)}; background: transparent;")


class MK1Terminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MK1 — Dual Branch Terminal")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        self._apply_palette()
        self._build()

    def _apply_palette(self):
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(BG))
        p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
        p.setColor(QPalette.ColorRole.Base, QColor(PANEL))
        p.setColor(QPalette.ColorRole.Text, QColor(TEXT))
        self.setPalette(p)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet(f"background: {PANEL}; border-bottom: 1px solid {BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)

        title = QLabel("MK1")
        title.setFont(QFont(SANS, 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_HI}; background: transparent;")
        hl.addWidget(title)

        sub = QLabel("Dual Branch Training Terminal")
        sub.setFont(QFont(MONO, 10))
        sub.setStyleSheet(f"color: {DIM}; background: transparent; margin-left: 12px;")
        hl.addWidget(sub)
        hl.addStretch()

        # Start both button
        start_all = styled_btn("▶ START BOTH", "#7c4dff", 120)
        start_all.clicked.connect(self._start_both)
        hl.addWidget(start_all)

        stop_all = styled_btn("■ STOP ALL", RED, 100)
        stop_all.clicked.connect(self._stop_both)
        hl.addWidget(stop_all)

        root.addWidget(header)

        # Split panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; width: 2px; }}")

        self.h_panel = BranchPanel("H", H_COLOR)
        self.a_panel = BranchPanel("A", A_COLOR)

        splitter.addWidget(self.h_panel)
        splitter.addWidget(self.a_panel)
        splitter.setSizes([700, 700])

        root.addWidget(splitter, 1)

        # Bottom status bar
        status = QWidget()
        status.setFixedHeight(28)
        status.setStyleSheet(f"background: {PANEL}; border-top: 1px solid {BORDER};")
        sl = QHBoxLayout(status)
        sl.setContentsMargins(12, 0, 12, 0)

        self.status_label = QLabel("MK1 Terminal Ready — Select a branch to start training")
        self.status_label.setFont(QFont(MONO, 9))
        self.status_label.setStyleSheet(f"color: {DIM}; background: transparent;")
        sl.addWidget(self.status_label)
        sl.addStretch()

        time_label = QLabel(datetime.now().strftime("%Y-%m-%d"))
        time_label.setFont(QFont(MONO, 9))
        time_label.setStyleSheet(f"color: {DIM}; background: transparent;")
        sl.addWidget(time_label)

        root.addWidget(status)

    def _start_both(self):
        self.h_panel.start_training()
        # Small delay so H starts first
        QTimer.singleShot(2000, self.a_panel.start_training)
        self.status_label.setText("Both branches training simultaneously")

    def _stop_both(self):
        self.h_panel.stop_training()
        self.a_panel.stop_training()
        self.status_label.setText("All training stopped")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    window = MK1Terminal()
    window.show()
    sys.exit(app.exec())
