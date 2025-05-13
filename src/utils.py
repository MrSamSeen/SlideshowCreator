import shutil
import time

# ===== TERMINAL FORMATTING =====

class Colors:
    """Terminal colors for formatted output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    terminal_width = shutil.get_terminal_size().columns
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * terminal_width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(terminal_width)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * terminal_width}{Colors.ENDC}\n")

def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.ENDC}")

def print_info(text):
    """Print an info message"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_waiting(text, seconds):
    """Print a waiting message with countdown"""
    for i in range(seconds, 0, -1):
        print(f"{Colors.CYAN}⏱ {text} ({i}s remaining)...{Colors.ENDC}", end='\r')
        time.sleep(1)
    print(" " * shutil.get_terminal_size().columns, end='\r')  # Clear the line