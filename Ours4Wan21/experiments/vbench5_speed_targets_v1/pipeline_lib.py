"""Pure calibration and artifact-validation helpers for the five-prompt suite."""
from __future__ import annotations

import json
from pathlib import Path

MODES = (
    'scalar5', 'sea7', 'sea7_dynamics_raw_sea128', 'sea7_cache_update192',
    'sea7_local_drift1024', 'sea7_spatial_gradient96', 'sea7_channel_geometry240',
    'sea7_distribution256', 'sea7_spectral_drift512', 'sea7_spectral_phase512',
    'sea7_spectral_shape576', 'sea7_spectral_dynamics1024',
)
SHORT = {
    'scalar5': 'Scalar5', 'sea7': 'SEA7',
    'sea7_dynamics_raw_sea128': 'Dynamics128',
    'sea7_cache_update192': 'CacheUpdate192',
    'sea7_local_drift1024': 'LocalDrift1024',
    'sea7_spatial_gradient96': 'SpatialGrad96',
    'sea7_channel_geometry240': 'ChannelGeom240',
    'sea7_distribution256': 'Distribution256',
    'sea7_spectral_drift512': 'SpectralDrift512',
    'sea7_spectral_phase512': 'SpectralPhase512',
    'sea7_spectral_shape576': 'SpectralShape576',
    'sea7_spectral_dynamics1024': 'SpectralDynamics1024',
}
TARGETS = (1.8, 2.4, 3.0)
TARGET_K = (23, 29, 35)


def target_label(value: float) -> str:
    return f'{value:.1f}x'.replace('.', 'p')


def condition_name(mode: str, skip_budget: int) -> str:
    return f'{mode}/K{skip_budget:02d}'


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def validate_generation(path: Path, *, prompt_ids: list[str], method: str,
                        mode: str | None = None, skip_budget: int | None = None) -> dict:
    manifest = read_json(path / 'run.json')
    complete = read_json(path / 'COMPLETE.json')
    observed_ids = [row['sample_id'] for row in manifest.get('prompts', [])]
    if observed_ids != prompt_ids or complete.get('videos') != len(prompt_ids):
        raise ValueError(f'incomplete or mismatched generation: {path}')
    if manifest.get('method') != method:
        raise ValueError(f'wrong generation method: {path}')
    if method == 'ours' and (manifest.get('state_mode') != mode or
                             manifest.get('skip_budget') != skip_budget):
        raise ValueError(f'candidate mode/K mismatch: {path}')
    expected = set(prompt_ids)
    for folder, suffix in (('videos', '.mp4'), ('timings', '.json'), ('traces', '.json')):
        observed = {item.stem for item in (path / folder).glob(f'*{suffix}')}
        if observed != expected:
            raise ValueError(f'{folder} set mismatch: {path}')
    return manifest


def vbench_enabled(root: Path) -> bool:
    """Persist the user's reduced evaluation scope across resumes."""
    path = root / 'VBENCH_SKIPPED_BY_USER.json'
    if not path.exists():
        return True
    marker = read_json(path)
    if marker.get('status') != 'skipped_by_user' or marker.get('video_metrics_enabled') is not True:
        raise ValueError('invalid user VBench scope override')
    return False


def validate_quality(path: Path, prompt_count: int, *, require_vbench: bool = True) -> None:
    marker = read_json(path / 'quality/COMPLETE.json')
    video = read_json(path / 'quality/video_metrics/summary.json')
    if marker.get('status') != 'quality_complete' or video.get('video_count') != prompt_count:
        raise ValueError(f'incomplete quality result: {path}')
    if video.get('frame_count_total') != prompt_count * 81:
        raise ValueError(f'wrong quality frame count: {path}')
    if not require_vbench:
        if marker.get('vbench_status') != 'skipped_by_user':
            raise ValueError(f'quality result lacks explicit VBench skip status: {path}')
        return
    vbench = read_json(path / 'quality/vbench_custom/vbench_custom_aggregate_scores.json')
    if vbench.get('official_full_vbench_score') is not False or len(
            vbench.get('raw_dimension_scores', {})) != 10:
        raise ValueError(f'wrong VBench custom-input scope: {path}')
