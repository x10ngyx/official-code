"""Strict RGB video decoding for the frozen full-reference protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode_video_rgb(path: str | Path) -> np.ndarray:
    """Decode a video as T,C,H,W float32 RGB in [0,1]."""

    video_path = Path(path).resolve(strict=True)
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    frames: list[np.ndarray] = []
    try:
        for index, frame in enumerate(reader):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    f"{video_path} frame {index} is not RGB H,W,3; got {array.shape}"
                )
            if array.dtype != np.uint8:
                raise TypeError(
                    f"{video_path} frame {index} must decode as uint8 RGB; got {array.dtype}"
                )
            frames.append(array)
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"{video_path} decoded to zero frames")
    first_shape = frames[0].shape
    for index, frame in enumerate(frames[1:], start=1):
        if frame.shape != first_shape:
            raise ValueError(
                f"{video_path} changes shape at frame {index}: {first_shape} -> {frame.shape}"
            )

    video_thwc = np.stack(frames)
    return np.ascontiguousarray(
        video_thwc.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    )
