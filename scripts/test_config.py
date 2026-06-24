import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config

config = load_config()
print("Config loaded successfully.")
print("Total PRB:", config["simulation"]["total_prb"])
print("Action values:", config["action"]["alpha_values"])
