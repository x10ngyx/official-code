#!/usr/bin/env python3
"""Optional TeaCache injection for the locked original Wan2.1 pipelines."""

from __future__ import annotations

import logging
import types
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator

from upstream.teacache_generate import (
    i2v_generate,
    t2v_generate,
    teacache_forward,
)


T2V_COEFFICIENTS = {
    ("1.3B", False): [
        2.39676752e03,
        -1.31110545e03,
        2.01331979e02,
        -8.29855975e00,
        1.37887774e-01,
    ],
    ("14B", False): [
        -5784.54975374,
        5449.50911966,
        -1811.16591783,
        256.27178429,
        -13.02252404,
    ],
    ("1.3B", True): [
        -5.21862437e04,
        9.23041404e03,
        -5.28275948e02,
        1.36987616e01,
        -4.99875664e-02,
    ],
    ("14B", True): [
        -3.03318725e05,
        4.90537029e04,
        -2.65530556e03,
        5.87365115e01,
        -3.15583525e-01,
    ],
}

I2V_COEFFICIENTS = {
    ("480P", False): [
        -3.02331670e02,
        2.23948934e02,
        -5.25463970e01,
        5.87348440e00,
        -2.01973289e-01,
    ],
    ("720P", False): [
        -114.36346466,
        65.26524496,
        -18.82220707,
        4.91518089,
        -0.23412683,
    ],
    ("480P", True): [
        2.57151496e05,
        -3.54229917e04,
        1.40286849e03,
        -1.35890334e01,
        1.32517977e-01,
    ],
    ("720P", True): [
        8.10705460e03,
        2.13393892e03,
        -3.72934672e02,
        1.66203073e01,
        -4.17769401e-02,
    ],
}


def _model_variant(task: str, checkpoint_dir: str) -> tuple[str, bool]:
    is_i2v = task == "i2v-14B"
    if is_i2v:
        for marker in ("480P", "720P"):
            if marker in checkpoint_dir:
                return marker, True
        raise ValueError(
            "official TeaCache I2V coefficients require '480P' or '720P' "
            "in the checkpoint path"
        )

    marker = "1.3B" if task == "t2v-1.3B" else "14B"
    if marker not in checkpoint_dir:
        raise ValueError(
            f"official TeaCache coefficients require {marker!r} in the checkpoint path"
        )
    return marker, False


def apply_teacache(
    pipeline: Any,
    *,
    task: str,
    checkpoint_dir: str,
    sample_steps: int,
    threshold: float,
    use_ret_steps: bool,
) -> None:
    """Bind the official TeaCache methods and state to one Wan2.1 pipeline."""

    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    if threshold < 0:
        raise ValueError("TeaCache threshold must be non-negative")

    variant, is_i2v = _model_variant(task, checkpoint_dir)
    coefficients = (
        I2V_COEFFICIENTS[(variant, use_ret_steps)]
        if is_i2v
        else T2V_COEFFICIENTS[(variant, use_ret_steps)]
    )

    pipeline.generate = types.MethodType(
        i2v_generate if is_i2v else t2v_generate,
        pipeline,
    )
    model = pipeline.model
    model.forward = types.MethodType(teacache_forward, model)
    model.enable_teacache = True
    model.cnt = 0
    model.num_steps = sample_steps * 2
    # The fitted proxy is not guaranteed to be non-negative.  Use a comparison
    # sentinel for the explicit zero-threshold diagnostic so the official
    # forward still runs while every block evaluation remains full-compute.
    model.teacache_configured_thresh = threshold
    model.teacache_thresh = float("-inf") if threshold == 0 else threshold
    model.accumulated_rel_l1_distance_even = 0
    model.accumulated_rel_l1_distance_odd = 0
    model.previous_e0_even = None
    model.previous_e0_odd = None
    model.previous_residual_even = None
    model.previous_residual_odd = None
    model.use_ref_steps = use_ret_steps
    model.coefficients = list(coefficients)
    model.ret_steps = (5 if use_ret_steps else 1) * 2
    model.cutoff_steps = sample_steps * 2 if use_ret_steps else sample_steps * 2 - 2

    logging.info(
        "Enabled official TeaCache method: task=%s variant=%s threshold=%s "
        "effective_threshold=%s use_ret_steps=%s",
        task,
        variant,
        threshold,
        model.teacache_thresh,
        use_ret_steps,
    )


@contextmanager
def patch_pipeline_construction(
    wan_module: Any,
    *,
    task: str,
    checkpoint_dir: str,
    sample_steps: int,
    threshold: float,
    use_ret_steps: bool,
) -> Iterator[None]:
    """Apply TeaCache immediately after the original Wan pipeline is built."""

    if threshold < 0:
        raise ValueError("TeaCache threshold must be non-negative")
    pipeline_class = wan_module.WanI2V if task == "i2v-14B" else wan_module.WanT2V
    original_init = pipeline_class.__init__

    @wraps(original_init)
    def patched_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        original_init(instance, *args, **kwargs)
        apply_teacache(
            instance,
            task=task,
            checkpoint_dir=checkpoint_dir,
            sample_steps=sample_steps,
            threshold=threshold,
            use_ret_steps=use_ret_steps,
        )

    pipeline_class.__init__ = patched_init
    try:
        yield
    finally:
        pipeline_class.__init__ = original_init
