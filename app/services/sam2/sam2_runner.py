from __future__ import annotations

import contextlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch

from app.services.sam2.sam2_schema import (
    Sam2FrameResult,
    Sam2RunResult,
    Sam2RunSummary,
    YoloPrompt,
    bbox_center,
    encode_binary_mask_rle,
    mask_to_metrics,
)
from app.services.sam2.sam2_visualization import write_mask_overlay_video


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "backend" / "storage" / "outputs" / "sam2" / "runs"
DEFAULT_PROMPT_SOURCE = "yolo_scissors_detection"


@dataclass(slots=True)
class ExtractedVideo:
    frame_dir: Path
    processed_to_original: dict[int, int]
    source_fps: float
    output_fps: float
    width: int
    height: int
    frame_count: int


def create_run_dir(runs_root: Path = DEFAULT_RUNS_ROOT, run_id: str | None = None) -> tuple[str, Path]:
    resolved_run_id = run_id or str(uuid.uuid4())
    run_dir = runs_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return resolved_run_id, run_dir


def choose_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def extract_strided_frames(video_path: Path, frame_dir: Path, frame_stride: int) -> ExtractedVideo:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}")

    frame_dir.mkdir(parents=True, exist_ok=True)
    processed_to_original: dict[int, int] = {}
    frame_index = 0
    processed_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                frame_path = frame_dir / f"{processed_index:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write extracted frame: {frame_path}")
                processed_to_original[processed_index] = frame_index
                processed_index += 1

            frame_index += 1
    finally:
        cap.release()

    if not processed_to_original:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    return ExtractedVideo(
        frame_dir=frame_dir,
        processed_to_original=processed_to_original,
        source_fps=source_fps,
        output_fps=max(source_fps / frame_stride, 1.0),
        width=width,
        height=height,
        frame_count=frame_index,
    )


def build_video_predictor(
    *,
    device: torch.device,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> Any:
    _ensure_local_sam2_on_path()

    from sam2.build_sam import (  # noqa: PLC0415
        build_sam2_video_predictor,
        build_sam2_video_predictor_hf,
    )

    if sam2_checkpoint and sam2_config:
        return build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=str(device),
        )

    model_id = sam2_model_id or os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-tiny")
    return build_sam2_video_predictor_hf(model_id, device=str(device))


def run_sam2_tracking(
    *,
    video_path: Path,
    yolo_prompt: YoloPrompt,
    frame_stride: int,
    run_id: str,
    run_dir: Path,
    device: torch.device,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> Sam2RunResult:
    extracted = extract_strided_frames(
        video_path=video_path,
        frame_dir=run_dir / "frames",
        frame_stride=frame_stride,
    )

    predictor = build_video_predictor(
        device=device,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_model_id=sam2_model_id,
    )

    points = np.array([yolo_prompt.point], dtype=np.float32)
    box = np.array(yolo_prompt.bbox, dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    prompt_processed_frame = _processed_frame_for_original(
        extracted.processed_to_original,
        yolo_prompt.frame_index,
    )
    frame_results: list[Sam2FrameResult] = []
    masks_by_processed_frame: dict[int, np.ndarray] = {}

    with torch.inference_mode(), autocast_context(device):
        inference_state = predictor.init_state(
            video_path=str(extracted.frame_dir),
            async_loading_frames=False,
        )
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_processed_frame,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
            normalize_coords=True,
        )

        for processed_frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            mask = _mask_for_object(object_ids, mask_logits, obj_id=1)
            mask_area, centroid, bbox = mask_to_metrics(mask)
            center = bbox_center(bbox)
            original_frame_idx = extracted.processed_to_original[int(processed_frame_idx)]
            masks_by_processed_frame[int(processed_frame_idx)] = mask

            frame_results.append(
                Sam2FrameResult(
                    frame_index=original_frame_idx,
                    processed_frame_index=int(processed_frame_idx),
                    mask_area=mask_area,
                    centroid=centroid,
                    bbox=bbox,
                    bbox_center=center,
                    trajectory_point=center,
                    blade_point=None,
                    confidence=None,
                    mask_rle=encode_binary_mask_rle(mask),
                )
            )

    output_video_path = run_dir / "sam2_mask_overlay.mp4"
    trajectory_metrics = compute_trajectory_metrics(frame_results)
    write_mask_overlay_video(
        frame_dir=extracted.frame_dir,
        frame_results=frame_results,
        masks_by_processed_frame=masks_by_processed_frame,
        output_path=output_video_path,
        fps=extracted.output_fps,
        trajectory_metrics=trajectory_metrics,
    )

    successful_masks = sum(1 for frame in frame_results if frame.mask_area > 0)
    mask_areas = [frame.mask_area for frame in frame_results]
    summary = Sam2RunSummary(
        processed_frames=len(frame_results),
        successful_masks=successful_masks,
        failure_frames=[frame.frame_index for frame in frame_results if frame.mask_area == 0],
        avg_mask_area=round(float(np.mean(mask_areas)), 3) if mask_areas else 0.0,
        output_video_path=str(output_video_path),
        trajectory_metrics=trajectory_metrics,
        tracking_quality_note=(
            "Mask may include hand, but trajectory remains usable because the tracked "
            "region follows the scissors movement."
        ),
    )

    result = Sam2RunResult(
        run_id=run_id,
        video_path=str(video_path),
        model="sam2+yolo",
        device=device.type,
        frame_stride=frame_stride,
        prompt_source=DEFAULT_PROMPT_SOURCE,
        initial_prompt=yolo_prompt,
        frames=frame_results,
        summary=summary,
        run_dir=run_dir,
    )

    write_json(run_dir / "raw.json", result.to_dict(include_masks=True))
    write_json(run_dir / "summary.json", result.to_dict(include_masks=False))
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_trajectory_metrics(frame_results: list[Sam2FrameResult]) -> dict[str, Any]:
    valid_frames = [
        frame for frame in frame_results if frame.trajectory_point is not None and frame.mask_area > 0
    ]
    points = np.array([frame.trajectory_point for frame in valid_frames], dtype=np.float32)
    frame_indices = [frame.frame_index for frame in valid_frames]

    base_payload: dict[str, Any] = {
        "point_source": "bbox_center",
        "valid_points": len(valid_frames),
        "lost_frames": [frame.frame_index for frame in frame_results if frame.mask_area == 0],
        "sudden_jumps": [],
        "is_usable": len(valid_frames) >= 2,
    }
    if len(valid_frames) == 0:
        return {
            **base_payload,
            "avg_perpendicular_distance": None,
            "max_left_deviation": None,
            "max_right_deviation": None,
            "horizontal_drift_range": None,
            "mean_step_distance": None,
            "max_step_distance": None,
            "fitted_line": None,
        }

    xs = points[:, 0]
    horizontal_drift = float(xs.max() - xs.min())

    if len(valid_frames) == 1:
        return {
            **base_payload,
            "avg_perpendicular_distance": 0.0,
            "max_left_deviation": 0.0,
            "max_right_deviation": 0.0,
            "horizontal_drift_range": round(horizontal_drift, 3),
            "mean_step_distance": 0.0,
            "max_step_distance": 0.0,
            "fitted_line": None,
        }

    mean_point = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - mean_point, full_matrices=False)
    direction = vh[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    centered = points - mean_point
    signed_distances = centered @ normal
    perpendicular_distances = np.abs(signed_distances)

    projections = centered @ direction
    line_start = mean_point + direction * projections.min()
    line_end = mean_point + direction * projections.max()

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    jump_threshold = max(float(np.median(steps) * 3.0), 50.0) if len(steps) else 50.0
    sudden_jumps = [
        {
            "from_frame": frame_indices[index],
            "to_frame": frame_indices[index + 1],
            "distance": round(float(step), 3),
        }
        for index, step in enumerate(steps)
        if float(step) > jump_threshold
    ]

    return {
        **base_payload,
        "sudden_jumps": sudden_jumps,
        "is_usable": len(valid_frames) >= 2 and not base_payload["lost_frames"],
        "avg_perpendicular_distance": round(float(perpendicular_distances.mean()), 3),
        "max_left_deviation": round(float(min(0.0, signed_distances.min())), 3),
        "max_right_deviation": round(float(max(0.0, signed_distances.max())), 3),
        "horizontal_drift_range": round(horizontal_drift, 3),
        "mean_step_distance": round(float(steps.mean()), 3) if len(steps) else 0.0,
        "max_step_distance": round(float(steps.max()), 3) if len(steps) else 0.0,
        "jump_threshold": round(jump_threshold, 3),
        "fitted_line": {
            "point": [round(float(value), 3) for value in mean_point.tolist()],
            "direction": [round(float(value), 6) for value in direction.tolist()],
            "start_point": [round(float(value), 3) for value in line_start.tolist()],
            "end_point": [round(float(value), 3) for value in line_end.tolist()],
        },
    }


def _mask_for_object(object_ids: list[int], mask_logits: torch.Tensor, obj_id: int) -> np.ndarray:
    object_id_list = [int(object_id) for object_id in object_ids]
    if obj_id not in object_id_list:
        height, width = mask_logits.shape[-2:]
        return np.zeros((height, width), dtype=bool)

    object_index = object_id_list.index(obj_id)
    mask = (mask_logits[object_index] > 0.0).detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(bool)


def _processed_frame_for_original(
    processed_to_original: dict[int, int],
    original_frame_index: int,
) -> int:
    for processed_frame_index, mapped_original in processed_to_original.items():
        if mapped_original == original_frame_index:
            return processed_frame_index
    raise RuntimeError(
        f"Prompt frame {original_frame_index} was not included in the extracted stride frames"
    )


def _ensure_local_sam2_on_path() -> None:
    local_sam2 = PROJECT_ROOT / "sam2"
    if local_sam2.is_dir():
        local_sam2_str = str(local_sam2)
        if local_sam2_str not in sys.path:
            sys.path.insert(0, local_sam2_str)
