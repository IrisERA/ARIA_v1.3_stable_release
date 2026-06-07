"""
ARIA — Main Entry Point
========================
Boot sequence. Everything starts here, in the right order.

Boot order (DO NOT CHANGE):
  1. Kill switch armed first — always
  2. Telemetry watchdog — thermal safety before anything else
  3. Permission gate — locked down by default
  4. Dispatch — hands come online
  5. (future) Voice input
  6. (future) Intent parser
  7. (future) Memory system
  8. (future) Autonomous scheduler

To run:
    python aria.py

To kill from another terminal:
    kill -SIGTERM <pid>
    OR press Ctrl+C
    OR call KS.kill("manual") from anywhere in code
"""

import logging
import signal
import sys
import time
import os

# ------------------------------------------------------------------
# Logging setup — do this before any imports that use log
# ------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/aria.log"),
    ]
)
log = logging.getLogger("ARIA")

# ------------------------------------------------------------------
# ARIA imports
# ------------------------------------------------------------------
from core.killswitch import KS
from modules.telemetry import TelemetryWatchdog
from modules.permissions import PermissionGate, PermLevel
from modules.dispatch import Dispatch


def boot():
    log.info("=" * 60)
    log.info("  ARIA — Autonomous Reasoning & Interaction Agent")
    log.info("  Version: 0.1.0 — Shell / Foundation Layer")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. ARM KILL SWITCH — first. always.
    # ------------------------------------------------------------------
    KS.arm()

    # Register Ctrl+C and SIGTERM → clean kill
    def handle_signal(sig, frame):
        log.info(f"Signal {sig} received")
        KS.kill("signal interrupt")

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ------------------------------------------------------------------
    # 2. TELEMETRY — thermal safety online
    # ------------------------------------------------------------------
    watchdog = TelemetryWatchdog()
    watchdog.start()

    # Give it one poll cycle to get initial readings
    time.sleep(2.5)
    snap = watchdog.latest()
    if snap:
        cpu_str = f"{snap.cpu_temp:.1f}°C" if snap.cpu_temp else "N/A"
        gpu_str = f"{snap.gpu_temp:.1f}°C" if snap.gpu_temp else "N/A"
        log.info(f"🌡️  Initial temps — CPU: {cpu_str} | GPU: {gpu_str} | RAM: {snap.ram_used_pct:.1f}%")

    # ------------------------------------------------------------------
    # 3. PERMISSION GATE — locked down
    # ------------------------------------------------------------------
    gate = PermissionGate()
    gate.enable(PermLevel.STANDARD)   # start at standard
    log.info(f"🔒 Permission gate: {gate.status()['max_level']}")

    # ------------------------------------------------------------------
    # 4. DISPATCH — hands online
    # ------------------------------------------------------------------
    dispatch = Dispatch(gate)

    # ------------------------------------------------------------------
    # 5. SYSTEM CHECK — run a quick self-test
    # ------------------------------------------------------------------
    log.info("🔧 Running self-test...")
    result = dispatch.run("get_system_info")
    if result.success:
        log.info(f"✅ Self-test passed: {result.output}")
    else:
        log.warning(f"⚠️  Self-test issue: {result.error}")

    # ------------------------------------------------------------------
    # 6. READY
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("  ✅ ARIA is online and ready")
    log.info("  📁 Logs: ./logs/")
    log.info("  🔴 Kill: Ctrl+C or KS.kill('reason')")
    log.info("=" * 60)

    # Return the core objects — future modules will use these
    return {
        "killswitch": KS,
        "watchdog":   watchdog,
        "gate":       gate,
        "dispatch":   dispatch,
    }


def demo(aria: dict):
    """
    Quick demo of the shell working.
    Replace this with your actual logic.
    """
    dispatch = aria["dispatch"]
    gate     = aria["gate"]

    log.info("--- Running demo actions ---")

    # Write a file to sandbox
    r = dispatch.run("write_file", target="hello.txt", data="ARIA shell is working!\n")
    log.info(f"write_file: {r}")

    # Read it back
    r = dispatch.run("read_file", target="hello.txt")
    log.info(f"read_file: {r.output.strip()}")

    # Try a blocked action — should be denied
    r = dispatch.run("format_drive", target="C:")
    log.info(f"format_drive (should be blocked): {r.error}")

    # Try an elevated action without approval — should be denied
    r = dispatch.run("install_package", target="something")
    log.info(f"install_package (should be blocked at STANDARD): {r.error}")


if __name__ == "__main__":
    aria = boot()

    # Run demo
    demo(aria)

    # Keep alive — in real use, your voice loop or scheduler goes here
    log.info("💤 Shell running... Ctrl+C to stop")
    try:
        while KS.is_alive():
            time.sleep(1)
    except SystemExit:
        pass

    log.info("ARIA shutdown complete.")
