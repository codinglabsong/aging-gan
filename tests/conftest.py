import sys
from pathlib import Path

# Add src directory to path for tests
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
