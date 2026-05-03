"""
Standalone scissors YOLO test API:
upload video, run local YOLO detection, serve annotated results.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = ROOT / "data" / "input"
DATA_OUTPUT = ROOT / "data" / "output"
FRONTEND_DIR = ROOT / "frontend"
YOLO_MODEL_PATH = ROOT / "models" / "best.pt"

load_dotenv(ROOT / ".env")

YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
DEFAULT_FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "1"))
ANGLE_JUMP_LIMIT_DEGREES = 35.0
LOW_CONFIDENCE_ANGLE_THRESHOLD = 0.35
MIN_BLADE_CONTOUR_AREA = 40.0
MIN_BLADE_ASPECT_RATIO = 2.0
DRAW_REJECTED_DEBUG = os.getenv("DRAW_REJECTED_DEBUG", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="Local Scissors YOLO Test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_INPUT.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
yolo_model: YOLO | None = None

app.mount("/media/output", StaticFiles(directory=str(DATA_OUTPUT)), name="output_media")


@app.get("/")
def serve_index() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index)


@app.get("/api/config/status")
def config_status() -> dict:
    return {
        "yolo_model_path": YOLO_MODEL_PATH.relative_to(ROOT).as_posix(),
        "yolo_model_present": YOLO_MODEL_PATH.is_file(),
        "confidence": YOLO_CONFIDENCE,
        "default_frame_stride": DEFAULT_FRAME_STRIDE,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        suffix = ".mp4"

    video_id = str(uuid.uuid4())
    dest = DATA_INPUT / f"{video_id}.mp4"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    dest.write_bytes(content)

    return {
        "video_id": video_id,
        "video_path": dest.relative_to(ROOT).as_posix(),
    }


class DetectRequest(BaseModel):
    video_id: str
    frame_stride: int | None = None


def get_yolo_model() -> YOLO:
    global yolo_model
    if not YOLO_MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"best.pt does not exist. Put your trained model at {YOLO_MODEL_PATH}",
        )
    if yolo_model is None:
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return yolo_model


def detect_scissors(frame: Any, model: YOLO) -> list[dict]:
    try:
        results = model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"YOLO inference error: {str(exc)[:300]}",
        ) from exc

    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = _result_names(result, model)
    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    confidence_values = boxes.conf.detach().cpu().numpy()
    class_values = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else [None] * len(boxes)

    predictions: list[dict] = []
    for bbox, conf, class_id in zip(xyxy_values, confidence_values, class_values):
        x1, y1, x2, y2 = [float(value) for value in bbox.tolist()]
        predictions.append(
            {
                "class": _class_name(names, class_id) or "scissors",
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
            }
        )

    return predictions


def pick_best_scissors(
    predictions: list[dict],
    frame_width: int,
    frame_height: int,
) -> tuple[dict | None, list[dict]]:
    frame_area = max(1.0, float(frame_width * frame_height))
    raw_debug: list[dict] = []
    class_names = [str(pred.get("class", "")).lower().strip() for pred in predictions]
    has_scissors_class = any("scissor" in class_name for class_name in class_names)
    candidates: list[dict] = []

    for pred in predictions:
        cls = str(pred.get("class", "")).lower().strip()
        conf = float(pred.get("confidence", 0.0))
        bbox = [float(value) for value in pred["bbox"]]
        w = max(0.0, float(pred.get("width", 0.0)))
        h = max(0.0, float(pred.get("height", 0.0)))
        area_ratio = (w * h) / frame_area
        aspect_ratio = h / max(w, 1e-6)
        accepted = conf >= YOLO_CONFIDENCE and (
            "scissor" in cls or not has_scissors_class
        )
        rejection_reason = None
        if conf < YOLO_CONFIDENCE:
            rejection_reason = "low_confidence"
        elif has_scissors_class and "scissor" not in cls:
            rejection_reason = "not_scissors"

        raw_debug.append(
            {
                "class": pred.get("class", ""),
                "confidence": round(conf, 6),
                "bbox": [round(v, 3) for v in bbox],
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect_ratio, 6),
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            }
        )

        if accepted:
            candidates.append(pred)

    best = max(candidates, key=lambda pred: float(pred.get("confidence", 0.0)), default=None)
    return best, raw_debug


@app.post("/api/yolo/detect")
def run_yolo_detect(body: DetectRequest) -> dict:
    video_id = body.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id required")

    frame_stride = body.frame_stride or DEFAULT_FRAME_STRIDE
    frame_stride = max(1, int(frame_stride))

    input_path = DATA_INPUT / f"{video_id}.mp4"
    if not input_path.is_file():
        raise HTTPException(status_code=404, detail=f"No uploaded video for video_id={video_id}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Invalid video dimensions")

    out_video_path = DATA_OUTPUT / f"{video_id}_local_yolo.mp4"
    tmp_video_path = DATA_OUTPUT / f"{video_id}_local_yolo.tmp.mp4"
    out_json_path = DATA_OUTPUT / f"{video_id}_local_yolo.json"

    writer = cv2.VideoWriter(
        str(tmp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Could not create output video writer")

    frames: list[dict] = []
    frame_index = 0
    detections_count = 0
    last_bbox: list[float] | None = None
    last_confidence = 0.0
    previous_angle: float | None = None
    model = get_yolo_model()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            status = "skipped"
            bbox = None
            confidence = 0.0
            all_detections_count = 0

            if frame_index % frame_stride == 0:
                predictions = detect_scissors(frame, model)
                picked, raw_predictions = pick_best_scissors(predictions, width, height)
                all_detections_count = len(predictions)

                if picked is not None:
                    bbox = [float(value) for value in picked["bbox"]]
                    confidence = float(picked.get("confidence", 0.0))
                    last_bbox = bbox
                    last_confidence = confidence
                    detections_count += 1
                    status = "detected"
                else:
                    status = "no_detection"
                    raw_predictions = raw_predictions
            else:
                raw_predictions = []
                if last_bbox is not None:
                    bbox = last_bbox
                    confidence = last_confidence
                    status = "reused"

            if DRAW_REJECTED_DEBUG and raw_predictions:
                for candidate in raw_predictions:
                    if candidate.get("accepted"):
                        continue
                    cand_bbox = candidate.get("bbox")
                    if not isinstance(cand_bbox, list) or len(cand_bbox) != 4:
                        continue
                    rx1, ry1, rx2, ry2 = map(int, cand_bbox)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (130, 130, 130), 1)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"scissors {confidence:.2f} [{status}]",
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

                start, end = extend_angle_line(angle_to_draw, center_x, center_y, width, height)
                cv2.line(frame, start, end, (0, 255, 0), 4)
                previous_angle = angle_to_draw

            frames.append(
                {
                    "frame_index": frame_index,
                    "selected_bbox": bbox,
                    "selected_confidence": round(confidence, 6),
                    "status": status,
                    "raw_predictions": raw_predictions,
                    "bbox": bbox,
                    "confidence": round(confidence, 6),
                    "all_detections_count": all_detections_count,
                }
            )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        if tmp_video_path.exists():
            tmp_video_path.unlink()
        raise HTTPException(status_code=500, detail="No frames were processed from the input video")

    # Only expose the final mp4 after the writer closed cleanly.
    # This avoids serving half-written/corrupt files when a run is interrupted.
    if out_video_path.exists():
        out_video_path.unlink()
    tmp_video_path.replace(out_video_path)

    payload = {
        "video_id": video_id,
        "frame_count": frame_index,
        "frame_stride": frame_stride,
        "detections_count": detections_count,
        "model_path": YOLO_MODEL_PATH.relative_to(ROOT).as_posix(),
        "confidence_threshold": YOLO_CONFIDENCE,
        "frames": frames,
    }

    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "video_id": video_id,
        "output_video_path": out_video_path.relative_to(ROOT).as_posix(),
        "json_path": out_json_path.relative_to(ROOT).as_posix(),
        "frame_count": frame_index,
        "frame_stride": frame_stride,
        "detections_count": detections_count,
        "annotated_video_url": f"/media/output/{video_id}_local_yolo.mp4",
    }


def _result_names(result: Any, model: YOLO) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def _class_name(names: dict[int, str], class_id: Any) -> str | None:
    if class_id is None:
        return None
    return names.get(int(class_id), str(int(class_id)))


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

