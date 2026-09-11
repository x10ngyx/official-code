"""Only the threshold operand changes; use the original corrected SeaCache state machine."""
from dataclasses import replace
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seacache import SeaCacheConfig, SeaCacheController

ENDPOINTS = ((.04,.24),(.08,.40),(.12,.60),(.16,.80),(.20,1.00))

def linear_path(start, end):
    if not 0 < start <= end or not math.isfinite(start+end):
        raise ValueError('invalid endpoints')
    return [start+(end-start)*s/49 for s in range(50)]

class ScheduledController(SeaCacheController):
    def __init__(self, path):
        if len(path)!=50 or any(not math.isfinite(v) or v<=0 for v in path):
            raise ValueError('need 50 positive finite thresholds')
        self.threshold_path = list(path)
        super().__init__(SeaCacheConfig(threshold=path[0], use_ret_steps=False))

    def plan_step(self, **kwargs):
        self.config = replace(self.config, threshold=self.threshold_path[kwargs['step_index']])
        result = super().plan_step(**kwargs)
        self.decisions[-1]['requested_threshold'] = self.config.threshold
        return result

    def summary(self):
        out = super().summary()
        out.pop('threshold')
        out.update(threshold_path=self.threshold_path, schedule='explicit_50_step_path')
        return out
