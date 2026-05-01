from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

from app.services.sam2.sam2_runner import (
    DEFAULT_RUNS_ROOT,
    choose_device,
    create_run_dir,
    run_sam2_tracking,
)
from app.services.sam2.sam2_schema import Sam2RunResult, YoloPrompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_ROBOFLOW_MODEL_ID = "alis-workspace-awjkp/scissors-egocentric-instant-1"


@dataclass(slots=True)
class YoloDetection:
    bbox: list[float]
    confidence: float
    class_name: str | None

    @property
    def point(self) -> list[float]:
        x1, y1, x2, y2 = self.bbox
        return [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ]


def track_scissors_with_sam2(
    *,
    video_path: str | Path,
    yolo_model: str | Path | None = None,
    yolo_source: str = "roboflow",
    frame_stride: int = 5,
    use_gpu: bool = True,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    run_id: str | None = None,
    yolo_confidence: float = 0.25,
    bbox_shrink: float = 0.65,
    yolo_scan_frames: int = 30,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> Sam2RunResult:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video_path does not exist: {video}")

    device = choose_device(use_gpu=use_gpu)
    resolved_run_id, run_dir = create_run_dir(Path(runs_root), run_id)

    prompt_frame, prompt_frame_index, detection = find_initial_yolo_prompt(
        video_path=video,
        yolo_source=yolo_source,
        yolo_model=yolo_model,
        device=device.type,
        confidence_threshold=yolo_confidence,
        frame_stride=frame_stride,
        max_scanned_frames=yolo_scan_frames,
    )

    original_detection = detection
    frame_height, frame_width = prompt_frame.shape[:2]
    detection = YoloDetection(
        bbox=shrink_bbox(detection.bbox, bbox_shrink, frame_width, frame_height),
        confidence=detection.confidence,
        class_name=detection.class_name,
    )
    prompt = YoloPrompt(
        frame_index=prompt_frame_index,
        point=[round(float(value), 3) for value in detection.point],
        bbox=[round(float(value), 3) for value in detection.bbox],
        confidence=round(float(detection.confidence), 6),
        class_name=detection.class_name,
    )
    save_yolo_prompt_debug(
        frame=prompt_frame,
        original_detection=original_detection,
        shrunken_detection=detection,
        prompt=prompt,
        output_path=run_dir / "yolo_prompt_debug.jpg",
    )
    save_yolo_debug_json(
        output_path=run_dir / "yolo_prompt_debug.json",
        yolo_source=yolo_source,
        prompt=prompt,
        original_detection=original_detection,
        bbox_shrink=bbox_shrink,
    )

    return run_sam2_tracking(
        video_path=video,
        yolo_prompt=prompt,
        frame_stride=frame_stride,
        run_id=resolved_run_id,
        run_dir=run_dir,
        device=device,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_model_id=sam2_model_id,
    )


def read_first_usable_frame(video_path: Path) -> tuple[Any, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is not None and frame.size > 0:
                return frame, frame_index
            frame_index += 1
    finally:
        cap.release()

    raise RuntimeError(f"No usable frames found in video: {video_path}")


def find_initial_yolo_prompt(
    *,
    video_path: Path,
    yolo_source: str,
    yolo_model: str | Path | None,
    device: str,
    confidence_threshold: float,
    frame_stride: int,
    max_scanned_frames: int,
) -> tuple[Any, int, YoloDetection]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    yolo_model_path: Path | None = None
    if yolo_source == "local":
        if yolo_model is None:
            raise ValueError("--yolo_model is required when --yolo_source local is used")
        yolo_model_path = Path(yolo_model).expanduser().resolve()
        if not yolo_model_path.is_file():
            raise FileNotFoundError(f"yolo_model does not exist: {yolo_model_path}")

    frame_index = 0
    scanned_frames = 0
    errors: list[str] = []

    try:
        while scanned_frames < max(1, max_scanned_frames):
            ok, frame = cap.read()
            if not ok:
                break

            if frame is None or frame.size == 0:
                frame_index += 1
                continue
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            try:
                if yolo_source == "local":
                    assert yolo_model_path is not None
                    detection = detect_best_scissors_local(
                        frame=frame,
                        yolo_model_path=yolo_model_path,
                        device=device,
                        confidence_threshold=confidence_threshold,
                    )
                else:
                    detection = detect_best_scissors_roboflow(
                        frame=frame,
                        confidence_threshold=confidence_threshold,
                    )
                return frame, frame_index, detection
            except RuntimeError as exc:
                errors.append(f"frame {frame_index}: {exc}")

            scanned_frames += 1
            frame_index += 1
    finally:
        cap.release()

    detail = "; ".join(errors[-5:]) if errors else "no usable frames were scanned"
    raise RuntimeError(
        "YOLO did not detect scissors in the scanned prompt frames "
        f"(stride={frame_stride}, scanned={scanned_frames}). Last errors: {detail}"
    )


def detect_best_scissors(
    *,
    frame: Any,
    yolo_model_path: Path,
    device: str,
    confidence_threshold: float,
) -> YoloDetection:
    return detect_best_scissors_local(
        frame=frame,
        yolo_model_path=yolo_model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def detect_best_scissors_local(
    *,
    frame: Any,
    yolo_model_path: Path,
    device: str,
    confidence_threshold: float,
) -> YoloDetection:
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(yolo_model_path))
    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        device=0 if device == "cuda" else "cpu",
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no inference result for the first frame")

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("YOLO did not detect scissors in the first usable frame")

    names = _result_names(result, model)
    candidates: list[YoloDetection] = []
    scissors_candidates: list[YoloDetection] = []

    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    confidence_values = boxes.conf.detach().cpu().numpy()
    class_values = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else [None] * len(boxes)

    for bbox, confidence, class_id in zip(xyxy_values, confidence_values, class_values):
        class_name = _class_name(names, class_id)
        detection = YoloDetection(
            bbox=[float(value) for value in bbox.tolist()],
            confidence=float(confidence),
            class_name=class_name,
        )
        candidates.append(detection)
        if class_name and "scissor" in class_name.lower():
            scissors_candidates.append(detection)

    pool = scissors_candidates or candidates
    if not pool:
        raise RuntimeError("YOLO did not return a usable scissors detection")

    return max(pool, key=lambda detection: detection.confidence)


def detect_best_scissors_roboflow(
    *,
    frame: Any,
    confidence_threshold: float,
) -> YoloDetection:
    from inference_sdk import InferenceHTTPClient  # noqa: PLC0415

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    model_id = os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_ROBOFLOW_MODEL_ID).strip()
    if not api_key:
        raise RuntimeError("Missing ROBOFLOW_API_KEY in .env")
    if not model_id:
        raise RuntimeError("Missing ROBOFLOW_MODEL_ID in .env")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        frame_path = Path(tmp_file.name)

    try:
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError("Could not write temp frame for Roboflow inference")

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        result = client.infer(str(frame_path), model_id=model_id)
    finally:
        if frame_path.exists():
            frame_path.unlink()

    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError("Roboflow YOLO did not detect scissors in the first usable frame")

    candidates: list[YoloDetection] = []
    scissors_candidates: list[YoloDetection] = []
    for prediction in predictions:
        class_name = str(prediction.get("class", "")).strip()
        confidence = float(prediction.get("confidence", 0.0))
        if confidence < confidence_threshold:
            continue

        bbox = _roboflow_xywh_to_xyxy(prediction)
        detection = YoloDetection(
            bbox=bbox,
            confidence=confidence,
            class_name=class_name or None,
        )
        candidates.append(detection)
        if "scissor" in class_name.lower():
            scissors_candidates.append(detection)

    pool = scissors_candidates or candidates
    if not pool:
        raise RuntimeError("Roboflow YOLO returned detections, but none passed the scissors filters")

    return max(pool, key=lambda detection: detection.confidence)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track scissors with SAM2 using a YOLO detection center as the prompt point."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument(
        "--yolo_source",
        choices=["roboflow", "local"],
        default="roboflow",
        help="YOLO source. Defaults to Roboflow using ROBOFLOW_* values from .env.",
    )
    parser.add_argument(
        "--yolo_model",
        default=None,
        help="Path to a trained YOLO .pt model. Required only with --yolo_source local.",
    )
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for SAM2 tracking.")
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use CUDA when available. This is enabled by default.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_false",
        dest="use_gpu",
        help="Force CPU execution for debugging only.",
    )
    parser.add_argument("--runs_root", default=str(DEFAULT_RUNS_ROOT), help="Root folder for run outputs.")
    parser.add_argument("--run_id", default=None, help="Optional run id. Defaults to a UUID.")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--bbox_shrink",
        type=float,
        default=0.65,
        help="Scale the YOLO bbox around its center before using it as the SAM2 box prompt.",
    )
    parser.add_argument(
        "--yolo_scan_frames",
        type=int,
        default=30,
        help="Number of stride-aligned frames to scan for the first YOLO scissors prompt.",
    )
    parser.add_argument("--sam2_checkpoint", default=None, help="Optional local SAM2 checkpoint path.")
    parser.add_argument("--sam2_config", default=None, help="Optional local SAM2 config name/path.")
    parser.add_argument(
        "--sam2_model_id",
        default=None,
        help="Optional Hugging Face SAM2 model id when checkpoint/config are not provided.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = track_scissors_with_sam2(
        video_path=args.video_path,
        yolo_model=args.yolo_model,
        yolo_source=args.yolo_source,
        frame_stride=args.stride,
        use_gpu=args.use_gpu,
        runs_root=args.runs_root,
        run_id=args.run_id,
        yolo_confidence=args.yolo_conf,
        bbox_shrink=args.bbox_shrink,
        yolo_scan_frames=args.yolo_scan_frames,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        sam2_model_id=args.sam2_model_id,
    )
    print(json.dumps(result.to_dict(include_masks=False), indent=2))


def _result_names(result: Any, model: Any) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def _class_name(names: dict[int, str], class_id: Any) -> str | None:
    if class_id is None:
        return None
    return names.get(int(class_id), str(int(class_id)))


def _roboflow_xywh_to_xyxy(prediction: dict[str, Any]) -> list[float]:
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])
    return [
        x - width / 2.0,
        y - height / 2.0,
        x + width / 2.0,
        y + height / 2.0,
    ]


def shrink_bbox(
    bbox: list[float],
    scale: float,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    if scale <= 0:
        raise ValueError("--bbox_shrink must be greater than 0")

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(1.0, (x2 - x1) * scale)
    height = max(1.0, (y2 - y1) * scale)

    return [
        max(0.0, cx - width / 2.0),
        max(0.0, cy - height / 2.0),
        min(float(frame_width - 1), cx + width / 2.0),
        min(float(frame_height - 1), cy + height / 2.0),
    ]


def save_yolo_prompt_debug(
    *,
    frame: Any,
    original_detection: YoloDetection,
    shrunken_detection: YoloDetection,
    prompt: YoloPrompt,
    output_path: Path,
) -> None:
    debug_frame = frame.copy()
    _draw_bbox(
        debug_frame,
        original_detection.bbox,
        color=(0, 0, 255),
        label=f"YOLO original {original_detection.confidence:.2f}",
    )
    _draw_bbox(
        debug_frame,
        shrunken_detection.bbox,
        color=(0, 255, 255),
        label="SAM2 box prompt",
    )

    point_x, point_y = map(int, prompt.point)
    cv2.circle(debug_frame, (point_x, point_y), 7, (255, 0, 0), -1)
    cv2.putText(
        debug_frame,
        f"SAM2 point frame={prompt.frame_index}",
        (max(0, point_x - 90), max(24, point_y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), debug_frame):
        raise RuntimeError(f"Could not write YOLO debug image: {output_path}")


def save_yolo_debug_json(
    *,
    output_path: Path,
    yolo_source: str,
    prompt: YoloPrompt,
    original_detection: YoloDetection,
    bbox_shrink: float,
) -> None:
    payload = {
        "yolo_source": yolo_source,
        "bbox_shrink": bbox_shrink,
        "original_detection": {
            "bbox": [round(float(value), 3) for value in original_detection.bbox],
            "point": [round(float(value), 3) for value in original_detection.point],
            "confidence": round(float(original_detection.confidence), 6),
            "class_name": original_detection.class_name,
        },
        "sam2_prompt": prompt.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _draw_bbox(
    frame: Any,
    bbox: list[float],
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
