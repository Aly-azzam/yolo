"""
Standalone Roboflow scissors YOLO test API:
upload video, run Roboflow detection, serve annotated results.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from inference_sdk import InferenceHTTPClient
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = ROOT / "data" / "input"
DATA_OUTPUT = ROOT / "data" / "output"
FRONTEND_DIR = ROOT / "frontend"

load_dotenv(ROOT / ".env")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_MODEL_ID = os.getenv(
    "ROBOFLOW_MODEL_ID",
    "alis-workspace-awjkp/scissors-egocentric-instant-1",
).strip()
ROBOFLOW_CONFIDENCE = float(os.getenv("ROBOFLOW_CONFIDENCE", "0.5"))
DEFAULT_FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "3"))
DRAW_REJECTED_DEBUG = os.getenv("DRAW_REJECTED_DEBUG", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="Roboflow Scissors YOLO Test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_INPUT.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
rf_client: InferenceHTTPClient | None = None

app.mount("/media/output", StaticFiles(directory=str(DATA_OUTPUT)), name="output_media")


@app.get("/")
def serve_index() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index)


@app.get("/api/config/status")
def config_status() -> dict:
    model_id = os.getenv("ROBOFLOW_MODEL_ID", ROBOFLOW_MODEL_ID).strip()
    return {
        "roboflow_api_key_present": bool(ROBOFLOW_API_KEY),
        "roboflow_model_id": model_id,
        "confidence": ROBOFLOW_CONFIDENCE,
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


def get_roboflow_client() -> InferenceHTTPClient:
    global rf_client
    if not ROBOFLOW_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing ROBOFLOW_API_KEY in .env",
        )
    if rf_client is None:
        rf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
    return rf_client


def call_roboflow(frame_path: Path) -> list[dict]:
    model_id = os.getenv("ROBOFLOW_MODEL_ID", ROBOFLOW_MODEL_ID).strip()
    if not model_id:
        raise HTTPException(
            status_code=500,
            detail="Missing ROBOFLOW_MODEL_ID in .env",
        )

    try:
        client = get_roboflow_client()
        result = client.infer(str(frame_path), model_id=model_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow inference error: {str(exc)[:300]}",
        ) from exc

    predictions = result.get("predictions", [])
    if isinstance(predictions, list):
        return predictions
    return []


def xywh_to_xyxy(pred: dict) -> list[float]:
    x = float(pred["x"])
    y = float(pred["y"])
    w = float(pred["width"])
    h = float(pred["height"])

    return [
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2,
    ]


def pick_best_scissors(
    predictions: list[dict],
    frame_width: int,
    frame_height: int,
) -> tuple[dict | None, list[dict]]:
    best: dict | None = None
    best_score = float("-inf")
    best_conf = 0.0
    best_area_ratio = 1.0
    frame_area = max(1.0, float(frame_width * frame_height))
    raw_debug: list[dict] = []

    for pred in predictions:
        cls = str(pred.get("class", "")).lower().strip()
        conf = float(pred.get("confidence", 0.0))
        bbox = xywh_to_xyxy(pred)
        w = max(0.0, float(pred.get("width", 0.0)))
        h = max(0.0, float(pred.get("height", 0.0)))
        area_ratio = (w * h) / frame_area
        aspect_ratio = h / max(w, 1e-6)
        accepted = True
        rejection_reason: str | None = None

        if "scissors" not in cls:
            accepted = False
            rejection_reason = "not_scissors"
        elif conf < ROBOFLOW_CONFIDENCE:
            accepted = False
            rejection_reason = "low_confidence"
        elif area_ratio > 0.18:
            accepted = False
            rejection_reason = "too_large"
        elif area_ratio < 0.002:
            accepted = False
            rejection_reason = "too_small"
        elif aspect_ratio < 1.2:
            accepted = False
            rejection_reason = "too_wide"

        score = conf
        if aspect_ratio >= 2.0:
            score += 0.15
        score -= area_ratio * 1.5

        raw_debug.append(
            {
                "class": pred.get("class", ""),
                "confidence": round(conf, 6),
                "bbox": [round(v, 3) for v in bbox],
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect_ratio, 6),
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "score": round(score, 6),
            }
        )

        if not accepted:
            continue

        choose_this = False
        if best is None or score > best_score:
            choose_this = True
        else:
            conf_close = abs(conf - best_conf) <= 0.05
            score_close = abs(score - best_score) <= 0.03
            if conf_close and score_close and area_ratio < best_area_ratio:
                choose_this = True
        if choose_this:
            best = pred
            best_score = score
            best_conf = conf
            best_area_ratio = area_ratio
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

    out_video_path = DATA_OUTPUT / f"{video_id}_roboflow_yolo.mp4"
    tmp_video_path = DATA_OUTPUT / f"{video_id}_roboflow_yolo.tmp.mp4"
    out_json_path = DATA_OUTPUT / f"{video_id}_roboflow_yolo.json"

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
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    frame_path = Path(tmp_file.name)
                try:
                    if not cv2.imwrite(str(frame_path), frame):
                        raise HTTPException(status_code=500, detail="Could not write temp frame")
                    predictions = call_roboflow(frame_path)
                finally:
                    if frame_path.exists():
                        frame_path.unlink()
                picked, raw_predictions = pick_best_scissors(predictions, width, height)
                all_detections_count = len(predictions)

                if picked is not None:
                    bbox = xywh_to_xyxy(picked)
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

    model_id = os.getenv("ROBOFLOW_MODEL_ID", ROBOFLOW_MODEL_ID).strip()
    if not model_id:
        raise HTTPException(status_code=500, detail="Missing ROBOFLOW_MODEL_ID in .env")

    payload = {
        "video_id": video_id,
        "frame_count": frame_index,
        "frame_stride": frame_stride,
        "detections_count": detections_count,
        "model_id": model_id,
        "confidence_threshold": ROBOFLOW_CONFIDENCE,
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
        "annotated_video_url": f"/media/output/{video_id}_roboflow_yolo.mp4",
    }