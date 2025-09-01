# pyminesweep

# PyMineSweep

A Python tool to manage and interact with the Minesweeper application window.

## Features
- Launch Minesweeper application
- Minimize all windows except Minesweeper for distraction-free gameplay
- Detect and return window coordinates (top-left, bottom-right)
- Check if Minesweeper is already running
- List all visible windows (for debugging)
- Kill existing Minesweeper processes
- DPI awareness for high-DPI displays

## Getting Started

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Basic usage:
   ```
   python main.py
   ```
   This will find an existing Minesweeper window and minimize all other windows.

## Command Line Options

- `--kill-existing`: Terminate any running Minesweeper processes
- `list_windows`: List all visible windows (useful for debugging)
- `startMine`: Launch a new instance of Minesweeper

## Requirements
See `requirements.txt` for the complete list of dependencies.

---

**Note:** This project is intended for Windows systems.
