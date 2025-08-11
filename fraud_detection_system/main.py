import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from gui.main_window import MainWindow
from config.settings import UI_CONFIG

def main():
    # Create main application
    root = tk.Tk()
    root.title("Smart Transaction Monitor - Fraud Detection System")
    root.geometry(UI_CONFIG['window_size'])
    
    # Set theme
    style = ttk.Style()
    style.theme_use(UI_CONFIG['theme'])
    
    # Create and run main window
    app = MainWindow(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()