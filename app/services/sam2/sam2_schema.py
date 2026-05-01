from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class YoloPrompt:
    frame_index: int
    point: list[float]
    bbox: list[float]
    confidence: float
    class_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["class_name"] is None:
            payload.pop("class_name")
        return payload


@dataclass(slots=True)
class Sam2FrameResult:
    frame_index: int
    mask_area: int
    centroid: list[float] | None
    bbox: list[int] | None
    bbox_center: list[float] | None = None
    trajectory_point: list[float] | None = None
    blade_point: list[float] | None = None
    confidence: float | None = None
    processed_frame_index: int | None = None
    mask_rle: dict[str, Any] | None = None

    def to_dict(self, include_mask: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["mask_bbox"] = payload["bbox"]
        payload["mask_centroid"] = payload["centroid"]
        if payload["confidence"] is None:
            payload.pop("confidence")
        if payload["blade_point"] is None:
            payload.pop("blade_point")
        if payload["processed_frame_index"] is None:
            payload.pop("processed_frame_index")
        if not include_mask or payload["mask_rle"] is None:
            payload.pop("mask_rle")
        return payload


@dataclass(slots=True)
class Sam2RunSummary:
    processed_frames: int
    successful_masks: int
    failure_frames: list[int]
    avg_mask_area: float
    output_video_path: str
    trajectory_metrics: dict[str, Any]
    tracking_quality_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Sam2RunResult:
    run_id: str
    video_path: str
    model: str
    device: str
    frame_stride: int
    prompt_source: str
    initial_prompt: YoloPrompt
    frames: list[Sam2FrameResult]
    summary: Sam2RunSummary
    run_dir: Path

    def to_dict(self, include_masks: bool = False) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_path": self.video_path,
            "model": self.model,
            "device": self.device,
            "frame_stride": self.frame_stride,
            "prompt_source": self.prompt_source,
            "initial_prompt": self.initial_prompt.to_dict(),
            "frames": [frame.to_dict(include_mask=include_masks) for frame in self.frames],
            "summary": self.summary.to_dict(),
        }


def mask_to_metrics(mask: np.ndarray) -> tuple[int, list[float] | None, list[int] | None]:
    ys, xs = np.where(mask)
    mask_area = int(xs.size)
    if mask_area == 0:
        return 0, None, None

    centroid = [round(float(xs.mean()), 3), round(float(ys.mean()), 3)]
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]
    return mask_area, centroid, bbox


def bbox_center(bbox: list[int] | None) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [
        round((float(x1) + float(x2)) / 2.0, 3),
        round((float(y1) + float(y2)) / 2.0, 3),
    ]


def encode_binary_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as uncompressed COCO-style RLE counts."""
    binary = np.asarray(mask, dtype=np.uint8)
    pixels = binary.flatten(order="F")
    counts: list[int] = []
    last_value = 0
    run_length = 0

    for value in pixels:
        value_int = int(value)
        if value_int == last_value:
            run_length += 1
            continue
        counts.append(run_length)
        run_length = 1
        last_value = value_int

    counts.append(run_length)
    return {
        "size": [int(binary.shape[0]), int(binary.shape[1])],
        "counts": counts,
    }
