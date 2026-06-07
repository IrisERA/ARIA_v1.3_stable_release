"""
ARIA — Build Script
====================
Packages ARIA into a single .exe file using PyInstaller.

Usage:
    python build.py

Output:
    dist/ARIA.exe   ← double click to run, pin to taskbar
"""

import subprocess
import sys
import os

def build():
    print("=" * 50)
    print("  ARIA Build Script")
    print("=" * 50)

    # Install PyInstaller if needed
    print("\n[1/3] Checking PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    # Install all dependencies
    print("\n[2/3] Installing dependencies...")
    deps = ["PyQt6", "psutil"]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)

    # Build the exe
    print("\n[3/3] Building ARIA.exe...")
    cmd = [
        "pyinstaller",
        "--onefile",           # single .exe file
        "--windowed",          # no console window
        "--name", "ARIA",
        "--add-data", f"core{os.pathsep}core",
        "--add-data", f"modules{os.pathsep}modules",
        "app.py"
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("  BUILD SUCCESSFUL")
        print(f"  Output: dist/ARIA.exe")
        print("  Right click -> Pin to taskbar")
        print("=" * 50)
    else:
        print("\nBuild failed — check errors above")

if __name__ == "__main__":
    build()
