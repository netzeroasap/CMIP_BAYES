from pathlib import Path

# Get the project root (parent of utils/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UTILS_DIR = PROJECT_ROOT / "utils"


# Export for easy access
__all__ = ['PROJECT_ROOT', 'DATA_DIR', 'UTILS_DIR']
