from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
ANGLE_JUMP_LIMIT_DEGREES = 35.0
LOW_CONFIDENCE_ANGLE_THRESHOLD = 0.35
MIN_BLADE_CONTOUR_AREA = 40.0
MIN_BLADE_ASPECT_RATIO = 2.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a vertical line that follows the scissors x-position detected by YOLO."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the YOLO .pt model. Defaults to models/best.pt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output mp4 path. Defaults to data/output/<video>_yolo_vertical_line.mp4.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help='Optional Ultralytics device value, for example "0" for GPU or "cpu".',
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional frame limit for quick tests.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = process_video_with_vertical_line(
        video_path=args.video_path,
        model_path=args.model,
        output_path=args.output,
        confidence=args.conf,
        device=args.device,
        max_frames=args.max_frames,
    )
    print(f"Wrote annotated video: {output_path}")


def process_video_with_vertical_line(
    *,
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
    confidence: float = 0.25,
    device: str | None = None,
    max_frames: int | None = None,
) -> Path:
    video = Path(video_path).expanduser().resolve()
    model_file = Path(model_path).expanduser().resolve()

    if not video.is_file():
        raise FileNotFoundError(f"video_path does not exist: {video}")
    if not model_file.is_file():
        raise FileNotFoundError(f"model does not exist: {model_file}")

    output = resolve_output_path(video, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp_avi = output.with_suffix(".tmp.avi")
    tmp_mp4 = output.with_suffix(".tmp.mp4")

    model = YOLO(str(model_file))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video}")

    writer = cv2.VideoWriter(
        str(tmp_avi),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    frame_index = 0
    detections_count = 0
    previous_angle: float | None = None
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            detection = detect_best_scissors(
                model=model,
                frame=frame,
                confidence=confidence,
                device=device,
            )
            if detection is not None:
                previous_angle = draw_scissors_bbox_and_direction_line(
                    frame,
                    detection,
                    frame_width,
                    frame_height,
                    previous_angle,
                )
                detections_count += 1

            writer.write(frame)
            frame_index += 1
            if frame_index == 1 or frame_index % 30 == 0:
                print(
                    f"Processed {frame_index} frames; detections={detections_count}",
                    flush=True,
                )
    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        if tmp_avi.exists():
            tmp_avi.unlink()
        raise RuntimeError("No frames were processed from the input video")

    try:
        transcode_avi_mjpeg_to_h264_mp4(input_avi=tmp_avi, output_mp4=tmp_mp4, fps=fps)
    finally:
        if tmp_avi.exists():
            tmp_avi.unlink()

    if output.exists():
        output.unlink()
    tmp_mp4.replace(output)

    print(
        f"Finished {frame_index} frames; detections={detections_count}",
        flush=True,
    )
    return output


def transcode_avi_mjpeg_to_h264_mp4(*, input_avi: Path, output_mp4: Path, fps: float) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if output_mp4.exists():
        output_mp4.unlink()

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_avi),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-r",
        str(fps),
        str(output_mp4),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or f"ffmpeg failed with code {completed.returncode}")

    if not output_mp4.is_file() or output_mp4.stat().st_size <= 0:
        raise RuntimeError("ffmpeg produced an empty output file")


def resolve_output_path(video: Path, output_path: str | Path | None) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()
    return DEFAULT_OUTPUT_DIR / f"{video.stem}_yolo_vertical_line.mp4"


def detect_best_scissors(
    *,
    model: YOLO,
    frame: Any,
    confidence: float,
    device: str | None,
) -> dict[str, Any] | None:
    predict_kwargs: dict[str, Any] = {
        "source": frame,
        "conf": confidence,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    if not results:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    names = _result_names(result, model)
    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    confidence_values = boxes.conf.detach().cpu().numpy()
    class_values = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else [None] * len(boxes)

    candidates: list[dict[str, Any]] = []
    scissors_candidates: list[dict[str, Any]] = []
    for bbox, score, class_id in zip(xyxy_values, confidence_values, class_values):
        class_name = _class_name(names, class_id)
        detection = {
            "bbox": [float(value) for value in bbox.tolist()],
            "confidence": float(score),
            "class_name": class_name,
        }
        candidates.append(detection)
        if class_name and "scissor" in class_name.lower():
            scissors_candidates.append(detection)

    pool = scissors_candidates or candidates
    if not pool:
        return None

    return max(pool, key=lambda candidate: candidate["confidence"])


def draw_scissors_bbox_and_direction_line(
    frame: Any,
    detection: dict[str, Any],
    frame_width: int,
    frame_height: int,
    previous_angle: float | None,
) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    confidence = float(detection["confidence"])
    label = detection["class_name"] or "scissors"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    bbox_angle = angle_from_points((x1, y1), (x2, y2))
    blade_angle = estimate_blade_angle_from_crop(frame, (x1, y1, x2, y2))
    if previous_angle is None:
        angle_to_draw = blade_angle if blade_angle is not None else bbox_angle
    elif blade_angle is None or confidence < LOW_CONFIDENCE_ANGLE_THRESHOLD:
        angle_to_draw = previous_angle
    else:
        jump = abs(angle_delta_degrees(previous_angle, blade_angle))
        if jump > ANGLE_JUMP_LIMIT_DEGREES:
            angle_to_draw = previous_angle
        else:
            angle_to_draw = smooth_angle(previous_angle, blade_angle)

    start, end = extend_angle_line(angle_to_draw, center_x, center_y, frame_width, frame_height)
    cv2.line(frame, start, end, (0, 255, 0), 4)
    return angle_to_draw


def estimate_blade_angle_from_crop(
    frame: Any,
    bbox: tuple[int, int, int, int],
) -> float | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(0, min(frame_height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    metal_lower = np.array([0, 0, 100])
    metal_upper = np.array([180, 80, 255])
    metal_mask = cv2.inRange(hsv, metal_lower, metal_upper)

    skin_lower = np.array([0, 30, 60])
    skin_upper = np.array([25, 180, 255])
    skin_mask_1 = cv2.inRange(hsv, skin_lower, skin_upper)
    skin_mask_2 = cv2.inRange(hsv, np.array([160, 30, 60]), np.array([180, 180, 255]))
    skin_mask = cv2.bitwise_or(skin_mask_1, skin_mask_2)
    mask = cv2.bitwise_and(metal_mask, cv2.bitwise_not(skin_mask))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = choose_longest_blade_contour(contours)
    if contour is None:
        return None

    points = contour.reshape(-1, 2).astype(np.float32)
    points[:, 0] += x1
    points[:, 1] += y1
    if len(points) < 2:
        return None

    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return normalize_angle_180(math.degrees(math.atan2(float(vy), float(vx))))


def choose_longest_blade_contour(contours: tuple[Any, ...]) -> Any | None:
    best_contour = None
    best_length = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= MIN_BLADE_CONTOUR_AREA:
            continue

        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        short_side = min(width, height)
        long_side = max(width, height)
        if short_side <= 0:
            continue

        aspect_ratio = long_side / short_side
        if aspect_ratio <= MIN_BLADE_ASPECT_RATIO:
            continue

        if long_side > best_length:
            best_contour = contour
            best_length = long_side

    return best_contour


def extend_angle_line(
    angle_degrees: float,
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
    scale: int = 2,
) -> tuple[tuple[int, int], tuple[int, int]]:
    length = max(frame_width, frame_height) * scale
    angle_radians = math.radians(angle_degrees)
    ux = math.cos(angle_radians)
    uy = math.sin(angle_radians)

    start = (int(center_x - ux * length), int(center_y - uy * length))
    end = (int(center_x + ux * length), int(center_y + uy * length))

    return start, end


def angle_from_points(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return normalize_angle_180(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def normalize_angle_180(angle: float) -> float:
    return angle % 180.0


def angle_delta_degrees(previous_angle: float, current_angle: float) -> float:
    return ((current_angle - previous_angle + 90.0) % 180.0) - 90.0


def smooth_angle(previous_angle: float, current_angle: float) -> float:
    delta = angle_delta_degrees(previous_angle, current_angle)
    return normalize_angle_180(previous_angle + 0.6 * delta)


def _result_names(result: Any, model: YOLO) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def _class_name(names: dict[int, str], class_id: Any) -> str | None:
    if class_id is None:
        return None
    return names.get(int(class_id), str(int(class_id)))


if __name__ == "__main__":
    main()
