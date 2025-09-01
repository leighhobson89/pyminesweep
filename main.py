import time
import sys
import subprocess
import pygetwindow as gw
import ctypes
import psutil
import os
import cv2
import numpy as np
import mss
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable

from board_detector import GridSquareDetector
import time

@dataclass
class GridSquare:
    """Represents a single square in the Minesweeper grid."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    center_x: int = 0  # Center x coordinate for clicking
    center_y: int = 0  # Center y coordinate for clicking
    state: str = 'hidden'  # 'hidden', 'revealed', 'flagged', 'mine'
    
    def __post_init__(self):
        """Calculate center coordinates after initialization."""
        # These will be updated when the grid is organized
        self.center_x = self.x
        self.center_y = self.y
    
    def __str__(self, row=None, col=None):
        """Return a formatted string showing cell information."""
        if row is not None and col is not None:
            return (
                f"CELL {row:02d}{col:02d}\n"
                f"CENTER POINT {self.center_x},{self.center_y}\n"
                f"STATE: {self.state.upper()}"
            )
        return f'({self.x},{self.y}):{self.state[0].upper()}'

class GameState:
    """Tracks the state of the Minesweeper game."""
    def __init__(self):
        self.game_over = False
        self.victory = False
        self.grid = None
        self.moves_made = 0
        self.start_time = time.time()
        
    def mark_game_over(self, victory: bool = False) -> None:
        """Mark the game as over, with optional victory status."""
        self.game_over = True
        self.victory = victory
        
    def increment_moves(self) -> None:
        """Increment the move counter."""
        self.moves_made += 1
        
    def get_elapsed_time(self) -> float:
        """Return the elapsed time since game start in seconds."""
        return time.time() - self.start_time


class GameGrid:
    """Manages the state of the Minesweeper game grid."""
    def __init__(self):
        self.squares: List[GridSquare] = []
        self.rows = 0
        self.cols = 0
        self.square_size = 0
        self.game_state = GameState()
        
    def add_square(self, x: int, y: int) -> None:
        """Add a new grid square if it doesn't already exist."""
        # Check if square already exists
        if not any(sq.x == x and sq.y == y for sq in self.squares):
            self.squares.append(GridSquare(x, y))
            self._update_dimensions()
            self._organize_grid()
    
    def _update_dimensions(self) -> None:
        """Update grid dimensions based on added squares."""
        if not self.squares:
            return
            
        # Get unique x and y coordinates
        xs = {sq.x for sq in self.squares}
        ys = {sq.y for sq in self.squares}
        
        # Calculate square size based on minimum distance between coordinates
        if len(self.squares) > 1:
            sorted_x = sorted(xs)
            sorted_y = sorted(ys)
            
            x_diffs = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
            y_diffs = [sorted_y[i+1] - sorted_y[i] for i in range(len(sorted_y)-1)]
            
            if x_diffs and y_diffs:
                self.square_size = min(min(x_diffs), min(y_diffs))
    
    def _organize_grid(self) -> None:
        """Organize squares into rows and columns based on their coordinates."""
        if not self.squares:
            return
            
        # Group squares by their y-coordinate (rows)
        rows = {}
        for square in self.squares:
            if square.y not in rows:
                rows[square.y] = []
            rows[square.y].append(square)
        
        # Sort rows by y-coordinate (top to bottom)
        sorted_rows = sorted(rows.items(), key=lambda item: item[0])
        
        # Sort each row by x-coordinate (left to right)
        self.grid = []
        for y, row_squares in sorted_rows:
            sorted_row = sorted(row_squares, key=lambda sq: sq.x)
            self.grid.append(sorted_row)
        
        # Update center coordinates for each square
        if len(self.grid) > 0 and len(self.grid[0]) > 0:
            # Calculate square dimensions based on the first row
            first_row = self.grid[0]
            if len(first_row) > 1:
                square_width = first_row[1].x - first_row[0].x
                square_height = (self.grid[1][0].y - self.grid[0][0].y) if len(self.grid) > 1 else square_width
                
                # Update center coordinates for all squares
                for row in self.grid:
                    for square in row:
                        square.center_x = square.x + square_width // 2
                        square.center_y = square.y + square_height // 2
        
        # Update dimensions based on organized grid
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
    
    def get_grid_representation(self) -> List[List[Optional[GridSquare]]]:
        """Return a 2D array representation of the grid."""
        if not hasattr(self, 'grid') or not self.grid:
            return []
        return self.grid
    
    def __str__(self) -> str:
        """Return a string representation of the grid with detailed cell information."""
        grid = self.get_grid_representation()
        if not grid:
            return "Empty grid"
            
        lines = []
        
        # Add grid dimensions
        lines.append(f"Grid Size: {len(grid)} rows x {len(grid[0])} columns")
        lines.append("=" * 40)
        
        # Add each cell's information
        for row_idx, row in enumerate(grid):
            for col_idx, square in enumerate(row):
                if square:
                    # Pass row and column indices to get the detailed string
                    lines.append(square.__str__(row=row_idx, col=col_idx) + "\n")
                else:
                    lines.append(f"CELL {row_idx:02d}{col_idx:02d} - INVALID\n")
                
        return "\n".join(lines)

# Flag to track if minimize_other_windows has been called
_minimize_called = False

# Enable DPI awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

def get_screen_scale():
    """Get the current screen DPI scaling factor"""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetDesktopWindow()
        hdc = user32.GetDC(hwnd)
        LOGPIXELSX = 88  # Magic number for DPI awareness
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        user32.ReleaseDC(hwnd, hdc)
        return dpi / 96.0  # 96 is the default DPI
    except:
        return 1.0  # Fallback to 100% scaling

def kill_minesweep_processes():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == 'minesweep.exe':
                proc.kill()
                print(f"Killed minesweep.exe process with PID {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def launch_exe():
    exe_path = os.path.join(os.path.dirname(__file__), "mineSweep.exe")
    proc = subprocess.Popen([exe_path])
    print(f"Launched {exe_path} with PID {proc.pid}")
    return proc

def find_minesweep_window(retries=10, delay=0.5):
    for _ in range(retries):
        windows = gw.getWindowsWithTitle("Minesweeper")
        if windows:
            return windows[0]
        time.sleep(delay)
    return None


def print_window_coordinates(window):
    """Print the window coordinates in a clean format"""
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    width = right - left
    height = bottom - top
    print("\n=== Minesweeper Window Coordinates ===")
    print(f"Top-Left:     ({left}, {top})")
    print(f"Bottom-Right: ({right}, {bottom})")
    print(f"Size:         {width}x{height}")
    print("=================================")


def list_all_windows():
    """List all visible windows with their titles and handles."""
    print("\n=== Listing all visible windows ===")
    for i, win in enumerate(gw.getAllWindows()):
        if win.title:  # Only show windows with titles
            print(f"{i}: Title='{win.title}' | Handle={hex(win._hWnd) if hasattr(win, '_hWnd') else 'N/A'}")
    print("===============================\n")
    sys.exit(0)


def is_minesweeper_running():
    """Check if Minesweeper is currently running"""
    print("\n=== Checking for Minesweeper process ===")
    found = False
    for proc in psutil.process_iter(['name', 'exe']):
        try:
            proc_name = proc.info['name']
            proc_exe = proc.info['exe']
            if proc_name and 'minesweep' in proc_name.lower():
                print(f"Found matching process: {proc_name} ({proc_exe})")
                found = True
            if proc_name and 'minesweeper' in proc_name.lower():
                print(f"Found matching process: {proc_name} ({proc_exe})")
                found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    print("=================================\n")
    return found

def minimize_other_windows(minesweeper_window):
    """
    Minimize all windows except the Minesweeper window.
    This function will only execute once per program run.
    """
    global _minimize_called
    
    if _minimize_called:
        print("Window minimization already performed, skipping...")
        return True
        
    try:
        # Get all visible windows
        all_windows = gw.getAllWindows()
        
        # Minimize all windows except the Minesweeper window
        for window in all_windows:
            try:
                if window != minesweeper_window and window.isActive and window.visible:
                    window.minimize()
            except Exception as e:
                print(f"Error minimizing window {window.title}: {e}")
                continue
        
        # Bring Minesweeper to front
        minesweeper_window.restore()
        minesweeper_window.activate()
        
        # Set the flag to indicate we've minimized windows
        _minimize_called = True
        return True
    except Exception as e:
        print(f"Error in minimize_other_windows: {e}")
        return False

def find_minesweeper_window():
    """
    Find and return the Minesweeper window if it exists.
    
    Returns:
        Optional[pygetwindow.Window]: The Minesweeper window if found, None otherwise
    """
    try:
        windows = gw.getWindowsWithTitle("Minesweeper")
        return windows[0] if windows else None
    except Exception as e:
        print(f"Error finding Minesweeper window: {e}")
        return None

def check_game_status(window, check_interval=0.5):
    """
    Check if a game is in progress by looking for the new game display.
    Continuously monitors the game and saves screenshots as currentGrab.png
    Also detects and counts grid squares in the game.
    
    Args:
        window: The Minesweeper window object
        check_interval: Time in seconds between status checks
        
    Returns:
        bool: Current game status (True if game is in progress, False otherwise)
    """
    try:
        # Initialize grid square detector and game grid if not already done
        if not hasattr(check_game_status, 'grid_detector'):
            template_path = os.path.join("resources", "gridSquare.png")
            check_game_status.grid_detector = GridSquareDetector(template_path)
            check_game_status.game_grid = GameGrid()
        
        # Get window position and size
        x, y, w, h = window.left, window.top, window.width, window.height
        
        # Take a screenshot of the window
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": y, "left": x, "width": w, "height": h}
            
            # Grab the data
            sct_img = sct.grab(monitor)
            
            # Convert to numpy array
            screenshot = np.array(sct_img)
            
            # Convert BGRA to BGR
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Save the screenshot for debugging
            cv2.imwrite('currentGrab.png', screenshot)
            
        # Load the new game template
        template_path = os.path.join("resources", "newGameDisplay.png")
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            print(f"Error: Could not load template image from {template_path}")
            return True  # Default to game in progress if template can't be loaded
            
        # Perform template matching for game status
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Detect grid squares
        check_game_status.grid_detector.reset()  # Reset previous matches
        grid_square_count = check_game_status.grid_detector.find_grid_squares(screenshot)
        
        # Get the matched locations and add them to our game grid
        if hasattr(check_game_status.grid_detector, 'matched_locations'):
            for x, y in check_game_status.grid_detector.matched_locations:
                check_game_status.game_grid.add_square(x, y)
        
        # Check board size and print appropriate message
        if grid_square_count in [81, 256, 480]:
            difficulty = {
                81: "Beginner",
                256: "Intermediate",
                480: "Advanced"
            }[grid_square_count]
            
            print(f"{difficulty} Board...{grid_square_count} squares...Starting...")
            # print("\nGame Grid State:")
            # print("-" * 80)
            # print(check_game_status.game_grid)
            # print("-" * 80)
            
            # Set up the game state
            check_game_status.game_grid.game_state.grid = check_game_status.game_grid
            
            # Start the main game loop
            main_loop(check_game_status.game_grid)
            
            return False  # Stop checking and proceed to next step
        
        # If we get here, the board size isn't recognized yet
        print(f"Detected {grid_square_count} squares. Waiting for valid board...")
        time.sleep(1)
        return True  # Continue checking
        
        
    except KeyboardInterrupt:
        print("\nStopped monitoring game status")
        return False
    except Exception as e:
        print(f"Error checking game status: {e}")
        return False

def main_loop(game_grid: GameGrid) -> None:
    """
    Main game loop that handles the Minesweeper game logic.
    
    Args:
        game_grid: The game grid containing all the squares and game state
    """
    print("\n=== Starting Main Game Loop ===")
    
    while not game_grid.game_state.game_over:
        try:
            # TODO: Add game logic here
            # 1. Update grid state from screen
            # 2. Count remaining hidden cells
            # 3. Analyze board state
            # 4. Make a move or determine game end
            
            # Placeholder: Just print the current grid state
            # print("\nCurrent Grid State:")
            # print("-" * 80)
            # print(game_grid)
            # print("-" * 80)
            
            # For now, just wait a bit and then exit the loop
            print("Game logic will be implemented here. Exiting for now...")
            game_grid.game_state.mark_game_over()
            
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            break
        except Exception as e:
            print(f"\nError in main game loop: {e}")
            break
    
    # Game over handling
    if game_grid.game_state.victory:
        print("\n🎉 Congratulations! You won! 🎉")
    else:
        print("\nGame over!")
    
    print(f"Moves made: {game_grid.game_state.moves_made}")
    print(f"Time elapsed: {game_grid.game_state.get_elapsed_time():.1f} seconds")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--kill-existing":
        kill_minesweep_processes()
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "list_windows":
        list_all_windows()
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "startMine":
        # Kill any existing Minesweeper instances first
        kill_minesweep_processes()
        # Launch new instance
        launch_exe()
        sys.exit(0)
    
    # Default behavior: find window and minimize others
    window = find_minesweeper_window()
    if not window:
        print("Minesweeper window not found.")
        sys.exit(1)
    
    # Print window info
    print(f"\n=== Minesweeper Window Found ===")
    print(f"Title: '{window.title}'")
    print(f"Position: ({window.left}, {window.top})")
    print(f"Size: {window.width}x{window.height}")
    print("============================")
    
    # Minimize all other windows
    print("\nMinimizing other windows...")
    if minimize_other_windows(window):
        print("Successfully minimized other windows.")
    else:
        print("Warning: Could not minimize all windows.")
    
    # Bring Minesweeper to front
    window.restore()
    window.activate()
    
    # Continuously monitor game status
    print("\nStarting game status monitor...")
    try:
        while True:
            if not check_game_status(window):
                break  # Exit loop when valid board is detected
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped monitoring game status")
