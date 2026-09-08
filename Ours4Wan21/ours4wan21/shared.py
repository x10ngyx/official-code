"""Configure sibling CLI helpers for this process's declared remote roots."""
import sys
from .contracts import OFFICIAL, WORKSPACE, EXP_ROOT


def benchmark():
    sys.path.insert(0, str(OFFICIAL / 'Wan21Benchmark'))
    import protocol
    protocol.ROOT, protocol.EXP_ROOT = WORKSPACE, EXP_ROOT
    # Set roots before sibling generation/metrics import constants by value.
    import metrics
    metrics.ROOT = WORKSPACE
    return metrics
