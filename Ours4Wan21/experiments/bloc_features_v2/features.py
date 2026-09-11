"""Experiment import for the shared offline/online compact feature implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ours4wan21.bloc_features import CompactHistory, DIMS, EPS, VERSION
