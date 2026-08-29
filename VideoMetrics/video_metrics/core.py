"""Frozen metric kernels for full-reference RGB video fidelity."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

cv2.setNumThreads(1)

if TYPE_CHECKING:
    import torch


PROTOCOL_ID = "rgb_full_reference_v1"
PSNR_METRIC_NAME = "psnr_rgb_db"
SSIM_METRIC_NAME = "ssim_rgb"
LPIPS_METRIC_NAME = "lpips_alex_v0_1_spatial"
PSNR_MSE_CAP_THRESHOLD = 1e-10
PSNR_CAP_DB = 100.0


def validate_video_tchw(video: np.ndarray, *, name: str = "video") -> np.ndarray:
    """Validate a protocol-normalized video in T,C,H,W float format."""

    array = np.asarray(video)
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape T,C,H,W; got {array.shape}")
    if array.shape[0] < 1:
        raise ValueError(f"{name} has no frames")
    if array.shape[1] != 3:
        raise ValueError(f"{name} must have exactly three RGB channels; got {array.shape[1]}")
    if array.shape[2] < 11 or array.shape[3] < 11:
        raise ValueError(f"{name} frames must be at least 11x11 for SSIM; got {array.shape[2:]}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point in [0,1]; got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{name} must lie in [0,1]; observed [{minimum}, {maximum}]")
    return np.ascontiguousarray(array, dtype=np.float32)


def validate_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = validate_video_tchw(reference, name="reference")
    candidate = validate_video_tchw(candidate, name="candidate")
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference and candidate must have identical T,C,H,W shapes; "
            f"got {reference.shape} and {candidate.shape}"
        )
    return reference, candidate


def psnr_per_frame(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return frozen-protocol RGB PSNR for each aligned frame."""

    reference, candidate = validate_pair(reference, candidate)
    values: list[float] = []
    for ref_frame, candidate_frame in zip(reference, candidate, strict=True):
        # Preserve the locked upstream float32 subtraction/mean behavior.
        mse = float(np.mean((ref_frame / 1.0 - candidate_frame / 1.0) ** 2))
        if mse < PSNR_MSE_CAP_THRESHOLD:
            values.append(PSNR_CAP_DB)
        else:
            values.append(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return np.asarray(values, dtype=np.float64)


def _ssim_plane(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Frozen-protocol SSIM for one H,W channel plane."""

    c1 = 0.01**2
    c2 = 0.03**2
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu_reference = cv2.filter2D(reference, -1, window)[5:-5, 5:-5]
    mu_candidate = cv2.filter2D(candidate, -1, window)[5:-5, 5:-5]
    mu_reference_sq = mu_reference**2
    mu_candidate_sq = mu_candidate**2
    mu_cross = mu_reference * mu_candidate
    sigma_reference_sq = cv2.filter2D(reference**2, -1, window)[5:-5, 5:-5] - mu_reference_sq
    sigma_candidate_sq = cv2.filter2D(candidate**2, -1, window)[5:-5, 5:-5] - mu_candidate_sq
    sigma_cross = cv2.filter2D(reference * candidate, -1, window)[5:-5, 5:-5] - mu_cross
    ssim_map = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_reference_sq + mu_candidate_sq + c1)
        * (sigma_reference_sq + sigma_candidate_sq + c2)
    )
    return float(ssim_map.mean())


def ssim_per_frame(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return RGB channel-mean SSIM for each aligned frame."""

    reference, candidate = validate_pair(reference, candidate)
    values: list[float] = []
    for ref_frame, candidate_frame in zip(reference, candidate, strict=True):
        channel_values = [
            _ssim_plane(ref_frame[channel], candidate_frame[channel])
            for channel in range(3)
        ]
        values.append(float(np.asarray(channel_values).mean()))
    return np.asarray(values, dtype=np.float64)


class LPIPSComputer:
    """Reusable LPIPS AlexNet spatial-map evaluator."""

    def __init__(self, device: str = "auto", batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError("LPIPS batch_size must be at least 1")

        import lpips
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"LPIPS device {device!r} requested but CUDA is unavailable")

        self.device = torch.device(device)
        self.batch_size = batch_size
        # The locked protocol uses AlexNet with spatial=True. Version 0.1 is
        # explicit because the source evaluator relies on the package default.
        self.model = lpips.LPIPS(
            net="alex",
            spatial=True,
            version="0.1",
            verbose=False,
        ).eval().to(self.device)

    def per_frame(self, reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        import torch

        reference, candidate = validate_pair(reference, candidate)
        reference_tensor = torch.from_numpy(reference).mul(2.0).sub(1.0)
        candidate_tensor = torch.from_numpy(candidate).mul(2.0).sub(1.0)
        values: list[float] = []
        with torch.inference_mode():
            for start in range(0, reference_tensor.shape[0], self.batch_size):
                stop = min(start + self.batch_size, reference_tensor.shape[0])
                ref_batch = reference_tensor[start:stop].to(self.device)
                candidate_batch = candidate_tensor[start:stop].to(self.device)
                spatial_maps = self.model.forward(ref_batch, candidate_batch)
                batch_values = spatial_maps.mean(dim=(1, 2, 3)).detach().cpu()
                values.extend(float(value) for value in batch_values)
        return np.asarray(values, dtype=np.float64)
