"""
ARIA — Desktop Application
===========================
PyQt6 native desktop UI. Run this instead of aria.py directly.
Packages into a .exe with PyInstaller.

Install:
    pip install PyQt6 psutil

Run:
    python app.py

Build .exe:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name ARIA app.py
"""

import sys
import time
import threading
import os
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFrame, QProgressBar,
    QSplitter, QScrollArea, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QSize
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon, QPixmap, QPainter,
    QLinearGradient, QBrush, QTextCursor, QAction
)

# ── Colours ────────────────────────────────────────────────────────────────
BG        = "#06080f"
PANEL     = "#0c1020"
PANEL2    = "#111827"
BORDER    = "#1e2d45"
ACCENT    = "#00bfff"
ACCENT2   = "#0077cc"
GREEN     = "#00e676"
RED       = "#ff1744"
YELLOW    = "#ffd600"
ORANGE    = "#ff6d00"
TEXT      = "#8ab8d4"
TEXT_HI   = "#e0f0ff"
DIM       = "#2a3f55"

FONT_MONO = "Consolas"
FONT_DISP = "Segoe UI"


# ── Boot ARIA core in background ───────────────────────────────────────────
class ARIACore(QThread):
    log_signal    = pyqtSignal(str, str)   # message, level
    telemetry_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)        # online/offline/warn

    def __init__(self):
        super().__init__()
        self.aria = None
        self.running = True

    def run(self):
        try:
            # Add ARIA dir to path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

            self.log_signal.emit("Initializing ARIA core...", "info")

            from core.killswitch import KS
            from modules.telemetry import TelemetryWatchdog
            from modules.permissions import PermissionGate, PermLevel
            from modules.dispatch import Dispatch
            import logging

            # Suppress console logging — UI handles display
            logging.disable(logging.CRITICAL)

            KS.arm()
            self.log_signal.emit("Kill switch ARMED", "ok")

            watchdog = TelemetryWatchdog()
            watchdog.start()
            self.log_signal.emit("Telemetry watchdog online", "ok")

            time.sleep(2)

            gate = PermissionGate()
            gate.enable(PermLevel.STANDARD)
            self.log_signal.emit("Permission gate: STANDARD", "ok")

            dispatch = Dispatch(gate)
            self.log_signal.emit("Dispatch ready", "ok")
            self.log_signal.emit("ARIA is online and ready", "ok")
            self.status_signal.emit("online")

            self.aria = {"ks": KS, "watchdog": watchdog, "gate": gate, "dispatch": dispatch}

            # Telemetry polling loop
            while self.running and KS.is_alive():
                snap = watchdog.latest()
                if snap:
                    self.telemetry_signal.emit({
                        "cpu_temp":  snap.cpu_temp,
                        "gpu_temp":  snap.gpu_temp,
                        "cpu_load":  snap.cpu_load,
                        "ram_pct":   snap.ram_used_pct,
                        "gpu_load":  snap.gpu_load,
                        "vram_pct":  snap.gpu_vram_pct,
                        "status":    snap.status,
                        "warnings":  snap.warnings,
                    })
                time.sleep(2)

        except Exception as e:
            self.log_signal.emit(f"Core error: {e}", "error")
            self.status_signal.emit("offline")

    def send_command(self, text: str):
        """Process a text command through dispatch."""
        if not self.aria:
            return "ARIA core not ready"
        dispatch = self.aria["dispatch"]

        # Simple command routing
        text = text.lower().strip()
        if text.startswith("write "):
            parts = text[6:].split(" to ", 1)
            if len(parts) == 2:
                result = dispatch.run("write_file", target=parts[1].strip(), data=parts[0].strip())
                return f"Wrote to {parts[1].strip()}" if result.success else result.error
        elif text.startswith("read "):
            result = dispatch.run("read_file", target=text[5:].strip())
            return result.output if result.success else result.error
        elif text in ("status", "info", "system"):
            result = dispatch.run("get_system_info")
            return result.output if result.success else result.error
        elif text in ("help", "?"):
            return "Commands: status | read <file> | write <text> to <file> | kill"
        elif text == "kill":
            self.aria["ks"].kill("user command")
            return "Kill switch triggered"
        else:
            return f"Unknown command: '{text}' — type 'help' for commands"

    def stop(self):
        self.running = False
        if self.aria:
            self.aria["ks"].kill("app closed")


# ── Custom widgets ─────────────────────────────────────────────────────────

def make_label(text, font=FONT_MONO, size=11, color=TEXT, bold=False):
    l = QLabel(text)
    f = QFont(font, size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    return l


def h_line():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


class GaugeBar(QWidget):
    def __init__(self, label, warn=75, kill=90):
        super().__init__()
        self.warn = warn
        self.kill = kill
        self._value = 0
        self.setFixedHeight(28)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.lbl = make_label(label, size=10, color=DIM)
        self.lbl.setFixedWidth(72)
        layout.addWidget(self.lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self._apply_style(GREEN)
        layout.addWidget(self.bar)

        self.val = make_label("0%", size=10, color=TEXT_HI)
        self.val.setFixedWidth(48)
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.val)

    def _apply_style(self, color):
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background: {PANEL2};
                border: 1px solid {BORDER};
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
        """)

    def set_value(self, v, suffix="%"):
        if v is None: return
        self._value = v
        self.bar.setValue(int(min(100, max(0, v))))
        self.val.setText(f"{v:.0f}{suffix}")
        if v >= self.kill:
            self._apply_style(RED)
            self.val.setStyleSheet(f"color: {RED}; background: transparent;")
        elif v >= self.warn:
            self._apply_style(YELLOW)
            self.val.setStyleSheet(f"color: {YELLOW}; background: transparent;")
        else:
            self._apply_style(GREEN)
            self.val.setStyleSheet(f"color: {TEXT_HI}; background: transparent;")


class TempCard(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: {PANEL2};
                border: 1px solid {BORDER};
                border-radius: 3px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self.title = make_label(label, size=9, color=DIM)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.value = make_label("--", FONT_DISP, size=26, color=GREEN, bold=True)
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value)

        self.unit = make_label("°C", size=10, color=DIM)
        self.unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.unit)

    def set_temp(self, t):
        if t is None:
            self.value.setText("--")
            return
        self.value.setText(f"{t:.1f}")
        if t >= 70:
            self.value.setStyleSheet(f"color: {RED}; background: transparent;")
        elif t >= 65:
            self.value.setStyleSheet(f"color: {YELLOW}; background: transparent;")
        else:
            self.value.setStyleSheet(f"color: {GREEN}; background: transparent;")


class LogPanel(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont(FONT_MONO, 10))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {PANEL2};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 3px;
                padding: 8px;
            }}
        """)

    def append_line(self, msg: str, level: str = "info"):
        colors = {
            "info":  TEXT,
            "ok":    GREEN,
            "warn":  YELLOW,
            "error": RED,
            "kill":  RED,
            "cmd":   ACCENT,
            "reply": TEXT_HI,
        }
        color = colors.get(level, TEXT)
        t = datetime.now().strftime("%H:%M:%S")
        self.append(
            f'<span style="color:{DIM};">{t}</span> '
            f'<span style="color:{color};">{msg}</span>'
        )
        self.moveCursor(QTextCursor.MoveOperation.End)


# ── Main Window ────────────────────────────────────────────────────────────

class ARIAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARIA")
        self.setMinimumSize(900, 640)
        self.resize(1100, 720)
        self._apply_palette()
        self._build_ui()
        self._start_core()
        self._setup_tray()

    # ── Palette ────────────────────────────────────────────────────────────
    def _apply_palette(self):
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window,     QColor(BG))
        p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
        p.setColor(QPalette.ColorRole.Base,       QColor(PANEL))
        p.setColor(QPalette.ColorRole.Text,       QColor(TEXT))
        self.setPalette(p)
        self.setStyleSheet(f"background: {BG};")

    # ── UI Build ───────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(h_line())

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #1e2d45; width: 1px; }")
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.setSizes([260, 840])
        root.addWidget(splitter, 1)

        root.addWidget(h_line())
        root.addWidget(self._build_input_bar())

    def _build_header(self):
        w = QWidget()
        w.setFixedHeight(56)
        w.setStyleSheet(f"background: {PANEL};")
        layout = QHBoxLayout(w)
        layout.setContentsMargins(20, 0, 20, 0)

        # Logo
        logo_row = QHBoxLayout()
        logo_row.setSpacing(12)

        dot = QLabel("⬡")
        dot.setFont(QFont(FONT_DISP, 18))
        dot.setStyleSheet(f"color: {ACCENT};")
        logo_row.addWidget(dot)

        name = make_label("ARIA", FONT_DISP, 18, TEXT_HI, bold=True)
        logo_row.addWidget(name)

        sub = make_label("v0.1.0", size=10, color=DIM)
        sub.setAlignment(Qt.AlignmentFlag.AlignBottom)
        logo_row.addWidget(sub)

        layout.addLayout(logo_row)
        layout.addStretch()

        # Status
        self.status_label = make_label("● BOOTING", size=11, color=YELLOW)
        layout.addWidget(self.status_label)

        layout.addSpacing(20)

        # Kill button
        self.kill_btn = QPushButton("■ KILL")
        self.kill_btn.setFont(QFont(FONT_MONO, 10))
        self.kill_btn.setFixedSize(90, 30)
        self.kill_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {RED};
                border: 2px solid {RED};
                border-radius: 2px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background: {RED};
                color: #000;
            }}
            QPushButton:pressed {{ background: #cc0033; }}
        """)
        self.kill_btn.clicked.connect(self._on_kill)
        layout.addWidget(self.kill_btn)

        return w

    def _build_left(self):
        w = QWidget()
        w.setFixedWidth(260)
        w.setStyleSheet(f"background: {PANEL};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Section title
        layout.addWidget(make_label("// TELEMETRY", size=9, color=ACCENT))
        layout.addWidget(h_line())

        # Temp cards
        temp_row = QHBoxLayout()
        self.cpu_card = TempCard("CPU TEMP")
        self.gpu_card = TempCard("GPU TEMP")
        temp_row.addWidget(self.cpu_card)
        temp_row.addWidget(self.gpu_card)
        layout.addLayout(temp_row)

        layout.addSpacing(4)

        # Gauges
        self.cpu_bar  = GaugeBar("CPU", warn=75, kill=95)
        self.ram_bar  = GaugeBar("RAM", warn=85, kill=95)
        self.gpu_bar  = GaugeBar("GPU", warn=80, kill=95)
        self.vram_bar = GaugeBar("VRAM", warn=85, kill=95)
        layout.addWidget(self.cpu_bar)
        layout.addWidget(self.ram_bar)
        layout.addWidget(self.gpu_bar)
        layout.addWidget(self.vram_bar)

        layout.addSpacing(8)
        layout.addWidget(h_line())
        layout.addWidget(make_label("// LIMITS", size=9, color=ACCENT))

        limits = [
            ("CPU KILL", "70°C", RED),
            ("GPU KILL", "70°C", RED),
            ("CPU WARN", "65°C", YELLOW),
            ("RAM KILL", "95%",  RED),
            ("RATE LIM", "30/min", YELLOW),
        ]
        for name, val, col in limits:
            row = QHBoxLayout()
            row.addWidget(make_label(name, size=10, color=DIM))
            row.addStretch()
            row.addWidget(make_label(val, size=10, color=col))
            layout.addLayout(row)

        layout.addSpacing(8)
        layout.addWidget(h_line())
        layout.addWidget(make_label("// PERMISSIONS", size=9, color=ACCENT))

        self.perm_label = make_label("STANDARD", size=11, color=GREEN, bold=True)
        layout.addWidget(self.perm_label)

        layout.addStretch()
        return w

    def _build_center(self):
        w = QWidget()
        w.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Chat / log area
        layout.addWidget(make_label("// ARIA CONSOLE", size=9, color=ACCENT))

        self.log = LogPanel()
        layout.addWidget(self.log, 1)

        return w

    def _build_input_bar(self):
        w = QWidget()
        w.setFixedHeight(52)
        w.setStyleSheet(f"background: {PANEL}; border-top: 1px solid {BORDER};")
        layout = QHBoxLayout(w)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(8)

        prompt = make_label(">", size=13, color=ACCENT)
        layout.addWidget(prompt)

        self.input = QLineEdit()
        self.input.setFont(QFont(FONT_MONO, 11))
        self.input.setPlaceholderText("type a command... (try: status, help, read hello.txt)")
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {PANEL2};
                color: {TEXT_HI};
                border: 1px solid {BORDER};
                border-radius: 2px;
                padding: 4px 10px;
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT2};
            }}
        """)
        self.input.returnPressed.connect(self._on_send)
        layout.addWidget(self.input, 1)

        send_btn = QPushButton("SEND")
        send_btn.setFont(QFont(FONT_MONO, 10))
        send_btn.setFixedSize(70, 34)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT2};
                color: #fff;
                border: none;
                border-radius: 2px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {ACCENT}; color: #000; }}
            QPushButton:pressed {{ background: #005599; }}
        """)
        send_btn.clicked.connect(self._on_send)
        layout.addWidget(send_btn)

        return w

    # ── Core + signals ─────────────────────────────────────────────────────
    def _start_core(self):
        self.core = ARIACore()
        self.core.log_signal.connect(self._on_log)
        self.core.telemetry_signal.connect(self._on_telemetry)
        self.core.status_signal.connect(self._on_status)
        self.core.start()

    def _setup_tray(self):
        # System tray icon
        px = QPixmap(16, 16)
        px.fill(QColor(ACCENT))
        icon = QIcon(px)

        self.tray = QSystemTrayIcon(icon, self)
        menu = QMenu()

        show_action = QAction("Show ARIA", self)
        show_action.triggered.connect(self.show)
        menu.addAction(show_action)

        kill_action = QAction("Kill ARIA", self)
        kill_action.triggered.connect(self._on_kill)
        menu.addAction(kill_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.setToolTip("ARIA — Running")
        self.tray.activated.connect(lambda r: self.show() if r == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        self.tray.show()

    # ── Slots ──────────────────────────────────────────────────────────────
    def _on_log(self, msg: str, level: str):
        self.log.append_line(msg, level)

    def _on_telemetry(self, data: dict):
        self.cpu_card.set_temp(data.get("cpu_temp"))
        self.gpu_card.set_temp(data.get("gpu_temp"))
        self.cpu_bar.set_value(data.get("cpu_load", 0))
        self.ram_bar.set_value(data.get("ram_pct", 0))
        self.gpu_bar.set_value(data.get("gpu_load") or 0)
        self.vram_bar.set_value(data.get("vram_pct") or 0)

        status = data.get("status", "OK")
        if status == "CRITICAL":
            self.log.append_line("THERMAL WARNING — approaching kill threshold", "warn")

    def _on_status(self, status: str):
        colors = {"online": GREEN, "offline": RED, "warn": YELLOW}
        labels = {"online": "● ONLINE", "offline": "● OFFLINE", "warn": "● WARNING"}
        self.status_label.setText(labels.get(status, "● UNKNOWN"))
        self.status_label.setStyleSheet(f"color: {colors.get(status, YELLOW)}; background: transparent;")

    def _on_send(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.log.append_line(f"> {text}", "cmd")

        # Run in thread so UI doesn't freeze
        def run():
            reply = self.core.send_command(text)
            self.log.append_line(f"  {reply}", "reply")

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def _on_kill(self):
        self.log.append_line("KILL SWITCH TRIGGERED", "kill")
        self._on_status("offline")
        self.core.stop()

    def _quit(self):
        self.core.stop()
        QApplication.quit()

    def closeEvent(self, event):
        # Minimise to tray instead of closing
        event.ignore()
        self.hide()
        self.tray.showMessage("ARIA", "Running in system tray", QSystemTrayIcon.MessageIcon.Information, 2000)


# ── Entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = ARIAWindow()
    window.show()
    sys.exit(app.exec())
