"""CNN+G1 retains Wan21 state layout; only Wan22 protocol/boundaries change."""
import math
import torch

STEPS = 50
FORCED = (0, 32, 49)
STAGE_STARTS = (0, 32)
DIM = 18439
PROTOCOL = dict(model='Wan2.2-T2V-A14B', width=832, height=480, frames=45,
                fps=16, steps=50, solver='dpm++', shift=12,
                cfg_low_high=[3., 4.], boundary=.875, seed=42,
                dit_compute_dtype='bfloat16', offload_model=True, t5_cpu=False,
                batch_size=1, fsdp=False, sequence_parallel=False, prompt_extension=False)
FEATURE = dict(version='ours22_cnn_G1_features_v1', input_shape=[16,12,60,104],
    input_quantization='float16_then_float32', direct_pool=[4,8,8],
    temporal_mean_pool=[8,8], temporal_variance_pool=[8,8], variance_correction=0,
    variance_before_spatial_pool=True,
    roles=['current_input', 'previous_input', 'last_recompute_input'],
    layout='all_roles_3d_then_each_role_mean_var_2d_then_sea7',
    latent_filter='none', stage_start='zero_latent_features_and_reset_history',
    history_reset_steps=[0,32], forced_steps=list(FORCED),
    normalizer='train_only_per_coordinate_population_mean_std_floor_1e-6',
    normalized_cache_dtype='float16', model_input_dtype='float32')
SCALARS = ('sea_adjacent_relative_l1', 'sea_accumulated_with_current',
           'cached_valid', 'step_fraction', 'skip_budget_fraction',
           'used_skips_fraction', 'consecutive_skips_fraction')


def budget(value):
    if type(value) is not int or not 0 <= value <= 47:
        raise ValueError('Wan22 exact skip budget K must be an integer in [0,47]')
    return value


def scalar_state(step, k, used, consecutive, cached_valid, adjacent=0., accumulated=0.):
    budget(k)
    if any(type(x) is not int for x in (step, used, consecutive)):
        raise ValueError('integer counters required')
    if not (0 <= step < 50 and 0 <= consecutive <= used <= min(step, k)):
        raise ValueError('invalid counters')
    if type(cached_valid) is not bool or cached_valid != (step not in STAGE_STARTS):
        raise ValueError('cache validity disagrees with expert boundary')
    if step in FORCED:
        adjacent = accumulated = 0.
    if not all(math.isfinite(x) and x >= 0 for x in (adjacent, accumulated)):
        raise ValueError('invalid SEA distances')
    return torch.tensor([adjacent, accumulated, float(cached_valid), step/49.,
                         k/50., used/50., consecutive/50.], dtype=torch.float32)
