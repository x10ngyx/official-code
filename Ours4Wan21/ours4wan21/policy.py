"""Strict checkpoint loading; FP32 deterministic local actor inference."""
import json
from pathlib import Path
import time

import torch

from .contracts import FORCED, MODEL_ROOT, PROTOCOL, observation, state_contract, state_names, under
from .local_iql import IQLModelConfig, PolicyNet, target_skip_budget_from_calibration
from .overhead import network_flops, summarize_calls


class Policy:
    def __init__(self, checkpoint, *, device='cuda', state_mode=None, allow_smoke=False):
        path = under(checkpoint, MODEL_ROOT)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('schema') != 'ours4wan21_iql_checkpoint_v1' or payload.get('protocol') != PROTOCOL:
            raise ValueError('requires an Ours4Wan21 checkpoint with the frozen protocol')
        mode = payload['state']['mode']
        if state_mode is not None and state_mode != mode:
            raise ValueError('requested state mode differs from checkpoint')
        if payload['state'] != state_contract(mode):
            raise ValueError('checkpoint state contract mismatch')
        if payload.get('smoke_only') and not allow_smoke:
            raise ValueError('CPU smoke checkpoint cannot be used for production inference')
        self.mode, self.device = mode, torch.device(device)
        cfg = IQLModelConfig(**payload['model_config'])
        if cfg != IQLModelConfig(input_dim=len(state_names(mode))):
            raise ValueError('checkpoint architecture differs from frozen local MLP')
        self.net = PolicyNet(cfg).to(self.device).eval()
        self.net.load_state_dict(payload['policy_net'], strict=True)
        self.normalizer = {k: v.to(self.device) for k, v in payload['normalizer'].items()}
        if set(self.normalizer) != {'mean', 'std'} or any(
                v.shape != (cfg.input_dim,) or not torch.isfinite(v).all() for v in self.normalizer.values()):
            raise ValueError('invalid checkpoint normalizer')
        if not (self.normalizer['std'] > 0).all() or not all(torch.isfinite(p).all() for p in self.net.parameters()):
            raise ValueError('invalid weights/normalizer')
        self.flops_profile = network_flops(self.net)
        self._event_pairs = []
        if self.device.type == 'cuda':
            # Materialize events outside measured inference, then reuse per video.
            with torch.cuda.device(self.device):
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                    self.net(torch.zeros((1, cfg.input_dim), device=self.device, dtype=torch.float32))
                for _ in range(48):
                    pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    for event in pair:
                        event.record(torch.cuda.current_stream(self.device))
                    self._event_pairs.append(pair)
                self._event_pairs[-1][1].synchronize()
        self.reset_measurements()

    def reset_measurements(self):
        self._measurements = []

    def overhead_summary(self):
        if self.device.type == 'cuda' and self._measurements:
            self._event_pairs[len(self._measurements)-1][1].synchronize()
        calls = []
        for index, row in enumerate(self._measurements):
            seconds = (self._event_pairs[index][0].elapsed_time(self._event_pairs[index][1])/1000.
                       if self.device.type == 'cuda' else None)
            calls.append(dict(call_index=index, **row, network_cuda_seconds=seconds))
        return summarize_calls(calls, self.flops_profile, self.device.type)

    @torch.no_grad()
    def choose(self, state):
        index = len(self._measurements)
        if index >= 48:
            raise RuntimeError('reset predictor measurements before a new video')
        start, end = self._event_pairs[index] if self.device.type == 'cuda' else (None, None)
        decision_started = time.perf_counter()
        # DiT autocast must not silently run this FP32 MLP in BF16.
        with torch.autocast(device_type=self.device.type, enabled=False):
            x = state.to(device=self.device, dtype=torch.float32)
            x = (x - self.normalizer['mean']) / self.normalizer['std']
            inputs = x.unsqueeze(0)
            if start is not None:
                start.record(torch.cuda.current_stream(self.device))
            network_started = time.perf_counter()
            logits = self.net(inputs)
            network_host = time.perf_counter() - network_started
            if end is not None:
                end.record(torch.cuda.current_stream(self.device))
            logits = logits[0]
            if not torch.isfinite(logits).all():
                raise RuntimeError('nonfinite policy output')
            action, probability = int(logits.argmax().item()), float(logits.softmax(-1)[1].item())
        self._measurements.append(dict(network_host_span_seconds=network_host,
            decision_wall_seconds=time.perf_counter()-decision_started))
        return action, probability


def resolve_budget(*, skip_budget=None, target_speedup=None, calibration=None):
    if (skip_budget is None) == (target_speedup is None):
        raise ValueError('choose explicit K or target speedup plus Wan21 calibration')
    if skip_budget is not None:
        if type(skip_budget) is not int or not 0 <= skip_budget <= 48:
            raise ValueError('Wan21 skip budget must be an integer in [0,48]')
        return skip_budget
    if calibration is None:
        raise ValueError('target speedup needs empirical Wan21 calibration; Wan22 mappings are not reused')
    payload = json.loads(Path(calibration).read_text())
    if (payload.get('schema') != 'ours4wan21_speed_to_k_v1' or payload.get('status') != 'calibrated'
            or payload.get('protocol') != PROTOCOL or payload.get('forced_steps') != list(FORCED)):
        raise ValueError('calibration is not a completed frozen-protocol Wan21 mapping')
    if any(type(e['skip_budget']) is not int or not 0 <= e['skip_budget'] <= 48 for e in payload['entries']):
        raise ValueError('invalid calibrated K')
    return target_skip_budget_from_calibration(target_speedup, payload['entries'])
