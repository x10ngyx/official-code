"""Strict Wan22 actor loading and separately measured CNN decision overhead."""
import time
import torch
from .contracts import DIM, FEATURE, PROTOCOL, FORCED, budget
from .model import ARCH, CNN, encode_normalized
from .shared import MODELS, PROJECT, under, read

SCHEMA = 'ours4wan22_cnn_G1_checkpoint_v1'
DEFAULT_CALIBRATION = PROJECT / 'configs/speed_to_k_full_generate_20260912.json'


def resolve_budget(skip_budget=None, target_speedup=None, calibration=None):
    if (skip_budget is None) == (target_speedup is None):
        raise ValueError('choose either K or target speedup plus calibration')
    if skip_budget is not None:
        if calibration is not None:
            raise ValueError('calibration is only used with target speedup')
        return budget(skip_budget)
    p = read(DEFAULT_CALIBRATION if calibration is None else calibration)
    if (p.get('schema') != 'ours4wan22_speed_to_k_v1' or p.get('status') != 'calibrated'
            or p.get('protocol') != PROTOCOL or p.get('forced_steps') != list(FORCED)):
        raise ValueError('calibration protocol mismatch')
    if 'supported_target_speedup' in p:
        low, high = p['supported_target_speedup']
        if not low <= target_speedup <= high:
            raise ValueError(f'calibration supports target speedup [{low}, {high}]')
    for entry in p['entries']:
        budget(entry['skip_budget'])
    from ours4wan21.local_iql import target_skip_budget_from_calibration
    return target_skip_budget_from_calibration(target_speedup,p['entries'])


class Policy:
    def __init__(self, checkpoint, device='cuda'):
        self.path = under(checkpoint, MODELS)
        p = torch.load(self.path, map_location='cpu', weights_only=False)
        for key, expected in dict(schema=SCHEMA, architecture=ARCH,
                feature_contract=FEATURE, protocol=PROTOCOL, group='G1', smoke_only=False).items():
            if p.get(key) != expected:
                raise ValueError('checkpoint contract mismatch: ' + key)
        self.device = torch.device(device)
        self.net = CNN().to(self.device).eval()
        self.net.load_state_dict(p['policy_net'], strict=True)
        self.normalizer = {k: v.to(self.device) for k,v in p['normalizer'].items()}
        if set(self.normalizer) != {'mean','std'} or any(
                v.shape != (DIM,) or not torch.isfinite(v).all() for v in self.normalizer.values()):
            raise ValueError('normalizer contract')
        if (self.normalizer['std'] < 1e-6).any() or not all(torch.isfinite(p).all() for p in self.net.parameters()):
            raise ValueError('invalid normalizer/weights')
        self.events = []
        if self.device.type == 'cuda':
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                self.net(torch.zeros(1,DIM,device=self.device))
            for _ in range(47):
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for event in pair:
                    event.record(torch.cuda.current_stream(self.device))
                self.events.append(pair)
            self.events[-1][-1].synchronize()
        self.reset_measurements()

    def reset_measurements(self):
        self.measurements = []

    @torch.no_grad()
    def choose(self, raw):
        if raw.shape != (DIM,) or len(self.measurements) >= 47:
            raise ValueError('state dimension or actor call count')
        began = time.perf_counter()
        index = len(self.measurements)
        start,end = self.events[index] if self.device.type == 'cuda' else (None,None)
        with torch.autocast(self.device.type, enabled=False):
            x = encode_normalized(raw.to(self.device).unsqueeze(0), self.normalizer).float()
            if start is not None:
                start.record(torch.cuda.current_stream(self.device))
            tick = time.perf_counter()
            logits = self.net(x)[0]
            host = time.perf_counter()-tick
            if end is not None:
                end.record(torch.cuda.current_stream(self.device))
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite actor')
            action, probability = int(logits.argmax().item()), float(logits.softmax(-1)[1].item())
        self.measurements.append(dict(network_host_span_seconds=host, decision_wall_seconds=time.perf_counter()-began))
        return action, probability

    def overhead_summary(self):
        if self.events and self.measurements:
            self.events[len(self.measurements)-1][-1].synchronize()
        calls = [dict(row, network_cuda_seconds=(self.events[i][0].elapsed_time(self.events[i][1])/1000.
                   if self.events else None)) for i,row in enumerate(self.measurements)]
        return dict(calls=calls, call_count=len(calls),
            network_cuda_seconds=sum(r['network_cuda_seconds'] for r in calls) if self.events else None,
            decision_wall_seconds=sum(r['decision_wall_seconds'] for r in calls),
            estimated_tflops=len(calls)*24580608/1e12, flops_per_call=24580608,
            flops_scope='2 FLOPs/MAC, formal CNN Conv/Linear only; excludes pooling, normalization, SiLU and feature/SEA operations',
            timing_scope='nested in DiT/full generate; do not add again')
