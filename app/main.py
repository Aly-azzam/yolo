"""
Standalone scissors YOLO test API:
upload video, run local YOLO detection, serve annotated results.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from backend.app.services.dtw_service import run_dtw
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = ROOT / "data" / "input"
DATA_OUTPUT = ROOT / "data" / "output"
STORAGE_ROOT = ROOT / "storage"
SCISSORS_LINE_RUNS = STORAGE_ROOT / "scissors_line_runs"
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
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
SCISSORS_LINE_RUNS.mkdir(parents=True, exist_ok=True)
yolo_model: YOLO | None = None

app.mount("/media/output", StaticFiles(directory=str(DATA_OUTPUT)), name="output_media")
app.mount("/storage", StaticFiles(directory=str(STORAGE_ROOT)), name="storage")


@app.get("/")
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(
            """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Scissors Compare</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; }
      label { display: block; margin-top: 16px; font-weight: 700; }
      button { margin-top: 20px; padding: 10px 16px; }
      #status { margin-top: 20px; font-weight: 700; }
      .videos { display: grid; gap: 20px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 24px; }
      .preview { grid-column: 1 / -1; }
      video { background: #111; width: 100%; }
      .matches { margin-top: 28px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
      th { background: #f4f4f4; }
      td:first-child, th:first-child { text-align: center; }
      tr:hover { background: #fafafa; }
      .match-button { margin: 0; padding: 5px 10px; }
      @media (max-width: 760px) { .videos { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <h1>Scissors Compare</h1>
    <p>Upload expert and learner videos, then run YOLO scissors-line processing on both.</p>
    <form id="compare-form">
      <label>Expert video</label>
      <input name="expert_video" type="file" accept="video/*" required />
      <label>Learner video</label>
      <input name="learner_video" type="file" accept="video/*" required />
      <button type="submit">Run</button>
    </form>
    <div id="status">No run yet.</div>
    <div id="videos" class="videos"></div>
    <div id="matches" class="matches"></div>
    <script>
      const form = document.getElementById("compare-form");
      const status = document.getElementById("status");
      const videos = document.getElementById("videos");
      const matches = document.getElementById("matches");
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        status.textContent = "Running...";
        videos.innerHTML = "";
        matches.innerHTML = "";
        const response = await fetch("/api/compare/run", {
          method: "POST",
          body: new FormData(form),
        });
        const payload = await response.json();
        if (!response.ok) {
          status.textContent = payload.detail || "Run failed.";
          return;
        }

        const cacheBust = Date.now();
        const dtwScore = payload.dtw?.normalized_distance;
        status.textContent = Number.isFinite(dtwScore)
          ? `Done. DTW normalized distance: ${dtwScore.toFixed(2)} deg`
          : "Done.";
        videos.innerHTML = `
          <section>
            <h2>Expert Output</h2>
            <video id="expert-video" controls src="${payload.expert_output_video_url}?t=${cacheBust}"></video>
          </section>
          <section>
            <h2>Learner Comparison Output</h2>
            <video id="learner-video" controls src="${payload.learner_output_video_url}?t=${cacheBust}"></video>
          </section>
          <section class="preview">
            <h2>DTW Aligned Preview</h2>
            <video controls src="${payload.dtw_aligned_preview_video_url}?t=${cacheBust}"></video>
          </section>
        `;
        renderMatches(payload);
      });

      function renderMatches(payload) {
        const dtwMatches = payload.dtw?.matches || [];
        if (!dtwMatches.length) {
          matches.innerHTML = "<h2>DTW Matches</h2><p>No DTW matches returned.</p>";
          return;
        }

        const shownMatches = dtwMatches.slice(0, 100);
        const rows = shownMatches.map((match, index) => `
          <tr>
            <td><button class="match-button" data-index="${index}">View</button></td>
            <td>${match.expert_index}</td>
            <td>${match.learner_index}</td>
            <td>${Number(match.expert_angle).toFixed(1)}</td>
            <td>${Number(match.learner_angle).toFixed(1)}</td>
            <td>${Number(match.angle_difference).toFixed(1)}</td>
          </tr>
        `).join("");

        matches.innerHTML = `
          <h2>DTW Matches</h2>
          <p>Showing ${shownMatches.length} of ${dtwMatches.length} matched frame pairs. Click View to jump both videos.</p>
          <table>
            <thead>
              <tr>
                <th>View</th>
                <th>Expert Frame</th>
                <th>Learner Frame</th>
                <th>Expert Angle</th>
                <th>Learner Angle</th>
                <th>Difference</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        `;

        matches.querySelectorAll("button[data-index]").forEach((button) => {
          button.addEventListener("click", () => {
            const match = shownMatches[Number(button.dataset.index)];
            seekVideosToMatch(match, payload);
          });
        });
      }

      function seekVideosToMatch(match, payload) {
        const expertVideo = document.getElementById("expert-video");
        const learnerVideo = document.getElementById("learner-video");
        const expertFps = payload.expert_fps || 30;
        const learnerFps = payload.learner_fps || 30;
        expertVideo.currentTime = match.expert_frame_index / expertFps;
        learnerVideo.currentTime = match.learner_frame_index / learnerFps;
      }
    </script>
  </body>
</html>
            """,
        )
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


class ProcessedVideoResult(BaseModel):
    output_video_path: str
    json_path: str
    frame_count: int
    frame_stride: int
    fps: float
    detections_count: int
    expert_reference_angle: float | None = None
    learner_frames: list[dict] = Field(default_factory=list)


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

    result = process_video_with_scissors_line(
        input_path=input_path,
        output_stem=f"{video_id}_local_yolo",
        frame_stride=frame_stride,
    )

    return {
        "video_id": video_id,
        "output_video_path": result.output_video_path,
        "json_path": result.json_path,
        "frame_count": result.frame_count,
        "frame_stride": result.frame_stride,
        "fps": result.fps,
        "detections_count": result.detections_count,
        "annotated_video_url": f"/media/output/{Path(result.output_video_path).name}",
    }


@app.post("/api/compare/run")
async def run_compare(
    expert_video: UploadFile = File(...),
    learner_video: UploadFile = File(...),
    frame_stride: int | None = None,
) -> dict:
    run_id = str(uuid.uuid4())
    stride = max(1, int(frame_stride or DEFAULT_FRAME_STRIDE))

    expert_input = await save_uploaded_video(expert_video, f"{run_id}_expert")
    learner_input = await save_uploaded_video(learner_video, f"{run_id}_learner")

    expert_result = process_video_with_scissors_line(
        input_path=expert_input,
        output_stem=f"{run_id}_expert_output",
        frame_stride=stride,
        output_dir=DATA_OUTPUT,
        output_video_name=f"{run_id}_expert_output.mp4",
        output_json_name=f"{run_id}_expert_lines.json",
    )
    expert_reference_angle = compute_expert_reference_angle(ROOT / expert_result.json_path)

    learner_result = process_video_with_scissors_line(
        input_path=learner_input,
        output_stem=f"{run_id}_learner_comparison_output",
        frame_stride=stride,
        expert_reference_angle=expert_reference_angle,
        output_dir=DATA_OUTPUT,
        output_video_name=f"{run_id}_learner_comparison_output.mp4",
        output_json_name=f"{run_id}_learner_comparison.json",
    )
    expert_frames = extract_valid_line_frames(ROOT / expert_result.json_path)
    learner_frames = extract_valid_line_frames(ROOT / learner_result.json_path)
    expert_angles = [frame["line_angle"] for frame in expert_frames]
    learner_angles = [frame["line_angle"] for frame in learner_frames]
    dtw_result = run_dtw(
        expert_angles,
        learner_angles,
        window_ratio=dtw_window_ratio(expert_angles, learner_angles),
    )
    dtw_result = add_frame_indices_to_dtw_matches(
        dtw_result,
        expert_frames=expert_frames,
        learner_frames=learner_frames,
    )
    run_dir = SCISSORS_LINE_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dtw_preview_path = run_dir / "dtw_aligned_preview.mp4"
    create_dtw_aligned_preview(
        expert_video_path=ROOT / expert_result.output_video_path,
        learner_video_path=ROOT / learner_result.output_video_path,
        dtw_matches=dtw_result["matches"],
        output_path=dtw_preview_path,
    )

    return {
        "run_id": run_id,
        "expert_output_video_url": output_media_url(ROOT / expert_result.output_video_path),
        "learner_output_video_url": output_media_url(ROOT / learner_result.output_video_path),
        "dtw_aligned_preview_video_url": storage_url(dtw_preview_path),
        "expert_json_url": output_media_url(ROOT / expert_result.json_path),
        "learner_json_url": output_media_url(ROOT / learner_result.json_path),
        "expert_fps": expert_result.fps,
        "learner_fps": learner_result.fps,
        "expert_reference_angle": expert_reference_angle,
        "learner_frames": learner_result.learner_frames,
        "dtw": dtw_result,
    }


async def save_uploaded_video(file: UploadFile, stem: str) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        suffix = ".mp4"

    dest = DATA_INPUT / f"{stem}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Empty file for {file.filename}")

    dest.write_bytes(content)
    return dest


def output_media_url(path: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(DATA_OUTPUT.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Path is not inside public output directory: {path}",
        ) from exc
    return f"/media/output/{relative_path.as_posix()}"


def storage_url(path: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(STORAGE_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Path is not inside public storage directory: {path}",
        ) from exc
    return f"/storage/{relative_path.as_posix()}"


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


def process_video_with_scissors_line(
    *,
    input_path: Path,
    output_stem: str,
    frame_stride: int,
    expert_reference_angle: float | None = None,
    output_dir: Path = DATA_OUTPUT,
    output_video_name: str | None = None,
    output_json_name: str | None = None,
) -> ProcessedVideoResult:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Invalid video dimensions")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / (output_video_name or f"{output_stem}.mp4")
    # OpenCV's MPEG-4 Part 2 MP4 (`mp4v`) often won't play in browsers.
    # Write MJPEG into AVI first, then transcode to H.264 MP4 with FFmpeg.
    tmp_avi_path = output_dir / f"{out_video_path.stem}.tmp.avi"
    tmp_mp4_path = output_dir / f"{out_video_path.stem}.tmp.mp4"
    out_json_path = output_dir / (output_json_name or f"{output_stem}.json")

    writer = cv2.VideoWriter(
        str(tmp_avi_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Could not create output video writer")

    frames: list[dict] = []
    learner_frames: list[dict] = []
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
            line_center = None
            line_angle = None
            angle_difference = None
            valid_line = False

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
                line_center = [round(center_x, 3), round(center_y, 3)]
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

                if expert_reference_angle is not None:
                    ref_start, ref_end = extend_angle_line(
                        expert_reference_angle,
                        center_x,
                        center_y,
                        width,
                        height,
                    )
                    cv2.line(frame, ref_start, ref_end, (255, 0, 0), 3)

                start, end = extend_angle_line(angle_to_draw, center_x, center_y, width, height)
                cv2.line(frame, start, end, (0, 255, 0), 4)
                previous_angle = angle_to_draw
                line_angle = round(float(angle_to_draw), 6)
                valid_line = True
                if expert_reference_angle is not None:
                    angle_difference = round(
                        angle_difference_degrees(angle_to_draw, expert_reference_angle),
                        6,
                    )
                    draw_angle_difference_arc(
                        frame,
                        center=(center_x, center_y),
                        learner_angle=angle_to_draw,
                        expert_reference_angle=expert_reference_angle,
                        angle_difference=angle_difference,
                    )
                    draw_learner_compare_text(
                        frame,
                        learner_angle=angle_to_draw,
                        expert_reference_angle=expert_reference_angle,
                        angle_difference=angle_difference,
                    )
                    learner_frames.append(
                        {
                            "frame_index": frame_index,
                            "learner_angle": line_angle,
                            "expert_reference_angle": round(float(expert_reference_angle), 6),
                            "angle_difference": angle_difference,
                        }
                    )

            frames.append(
                {
                    "frame_index": frame_index,
                    "selected_bbox": bbox,
                    "selected_confidence": round(confidence, 6),
                    "bbox": bbox,
                    "line_center": line_center,
                    "line_angle": line_angle,
                    "expert_reference_angle": (
                        round(float(expert_reference_angle), 6)
                        if expert_reference_angle is not None and valid_line
                        else None
                    ),
                    "angle_difference": angle_difference,
                    "confidence": round(confidence, 6),
                    "valid_line": valid_line,
                    "status": status,
                    "raw_predictions": raw_predictions,
                    "all_detections_count": all_detections_count,
                }
            )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        if tmp_avi_path.exists():
            tmp_avi_path.unlink()
        raise HTTPException(status_code=500, detail="No frames were processed from the input video")

    try:
        transcode_avi_mjpeg_to_h264_mp4(
            input_avi=tmp_avi_path,
            output_mp4=tmp_mp4_path,
            fps=float(fps),
        )
    except RuntimeError as exc:
        if tmp_mp4_path.exists():
            tmp_mp4_path.unlink()
        if tmp_avi_path.exists():
            tmp_avi_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Could not encode browser-playable MP4: {str(exc)[:300]}",
        ) from exc

    if tmp_avi_path.exists():
        tmp_avi_path.unlink()

    # Only expose the final mp4 after encoding finished cleanly.
    if out_video_path.exists():
        out_video_path.unlink()
    tmp_mp4_path.replace(out_video_path)

    payload = {
        "input_video": input_path.relative_to(ROOT).as_posix(),
        "frame_count": frame_index,
        "frame_stride": frame_stride,
        "detections_count": detections_count,
        "model_path": YOLO_MODEL_PATH.relative_to(ROOT).as_posix(),
        "confidence_threshold": YOLO_CONFIDENCE,
        "expert_reference_angle": (
            round(float(expert_reference_angle), 6)
            if expert_reference_angle is not None
            else None
        ),
        "learner_frames": learner_frames,
        "frames": frames,
    }

    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return ProcessedVideoResult(
        output_video_path=out_video_path.relative_to(ROOT).as_posix(),
        json_path=out_json_path.relative_to(ROOT).as_posix(),
        frame_count=frame_index,
        frame_stride=frame_stride,
        fps=round(float(fps), 6),
        detections_count=detections_count,
        expert_reference_angle=(
            round(float(expert_reference_angle), 6)
            if expert_reference_angle is not None
            else None
        ),
        learner_frames=learner_frames,
    )


def _result_names(result: Any, model: YOLO) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def _class_name(names: dict[int, str], class_id: Any) -> str | None:
    if class_id is None:
        return None
    return names.get(int(class_id), str(int(class_id)))


def compute_expert_reference_angle(expert_json_path: Path) -> float:
    payload = json.loads(expert_json_path.read_text(encoding="utf-8"))
    angles = [
        float(frame["line_angle"])
        for frame in payload.get("frames", [])
        if frame.get("valid_line") and frame.get("line_angle") is not None
    ]
    if not angles:
        raise HTTPException(
            status_code=422,
            detail="Expert video did not produce any valid scissors line angles",
        )
    return round(float(statistics.median(angles)), 6)


def extract_valid_line_frames(json_path: Path) -> list[dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames = [
        {
            "frame_index": int(frame["frame_index"]),
            "line_angle": float(frame["line_angle"]),
        }
        for frame in payload.get("frames", [])
        if frame.get("valid_line") and frame.get("line_angle") is not None
    ]
    if not frames:
        raise HTTPException(
            status_code=422,
            detail=f"No valid line angles found in {json_path.name}",
        )
    return frames


def add_frame_indices_to_dtw_matches(
    dtw_result: dict,
    *,
    expert_frames: list[dict],
    learner_frames: list[dict],
) -> dict:
    enriched_matches = []
    for match in dtw_result["matches"]:
        expert_frame = expert_frames[match["expert_index"]]
        learner_frame = learner_frames[match["learner_index"]]
        enriched_matches.append(
            {
                **match,
                "expert_frame_index": expert_frame["frame_index"],
                "learner_frame_index": learner_frame["frame_index"],
            }
        )

    return {
        **dtw_result,
        "matches": enriched_matches,
    }


def create_dtw_aligned_preview(
    *,
    expert_video_path: Path,
    learner_video_path: Path,
    dtw_matches: list[dict],
    output_path: Path,
    fps: float = 10.0,
) -> None:
    if not dtw_matches:
        raise HTTPException(status_code=422, detail="DTW returned no matches for preview")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_avi_path = output_path.with_suffix(".tmp.avi")
    tmp_mp4_path = output_path.with_suffix(".tmp.mp4")

    expert_cap = cv2.VideoCapture(str(expert_video_path))
    learner_cap = cv2.VideoCapture(str(learner_video_path))
    if not expert_cap.isOpened() or not learner_cap.isOpened():
        expert_cap.release()
        learner_cap.release()
        raise HTTPException(status_code=500, detail="Could not open annotated videos for DTW preview")

    writer = None
    written_frames = 0
    try:
        for step_index, match in enumerate(dtw_matches):
            expert_frame = read_video_frame(expert_cap, int(match["expert_frame_index"]))
            learner_frame = read_video_frame(learner_cap, int(match["learner_frame_index"]))
            if expert_frame is None or learner_frame is None:
                continue

            combined = make_dtw_preview_frame(
                expert_frame=expert_frame,
                learner_frame=learner_frame,
                match=match,
                step_index=step_index,
            )
            if writer is None:
                height, width = combined.shape[:2]
                writer = cv2.VideoWriter(
                    str(tmp_avi_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise HTTPException(status_code=500, detail="Could not create DTW preview writer")

            writer.write(combined)
            written_frames += 1
    finally:
        expert_cap.release()
        learner_cap.release()
        if writer is not None:
            writer.release()

    if written_frames == 0:
        if tmp_avi_path.exists():
            tmp_avi_path.unlink()
        raise HTTPException(status_code=500, detail="No DTW preview frames were written")

    try:
        transcode_avi_mjpeg_to_h264_mp4(input_avi=tmp_avi_path, output_mp4=tmp_mp4_path, fps=fps)
    except RuntimeError as exc:
        if tmp_avi_path.exists():
            tmp_avi_path.unlink()
        if tmp_mp4_path.exists():
            tmp_mp4_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Could not encode DTW preview MP4: {str(exc)[:300]}",
        ) from exc

    if tmp_avi_path.exists():
        tmp_avi_path.unlink()
    if output_path.exists():
        output_path.unlink()
    tmp_mp4_path.replace(output_path)


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> Any | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def make_dtw_preview_frame(
    *,
    expert_frame: Any,
    learner_frame: Any,
    match: dict,
    step_index: int,
) -> Any:
    target_height = min(expert_frame.shape[0], learner_frame.shape[0])
    expert_frame = resize_to_height(expert_frame, target_height)
    learner_frame = resize_to_height(learner_frame, target_height)
    combined = np.hstack([expert_frame, learner_frame])

    overlay_lines = [
        f"DTW step: {step_index}",
        f"Expert frame: {match['expert_frame_index']}",
        f"Learner frame: {match['learner_frame_index']}",
        f"Expert angle: {float(match['expert_angle']):.1f} deg",
        f"Learner angle: {float(match['learner_angle']):.1f} deg",
        f"Difference: {float(match['angle_difference']):.1f} deg",
    ]
    y = 30
    for line in overlay_lines:
        cv2.putText(
            combined,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        y += 30

    split_x = expert_frame.shape[1]
    cv2.line(combined, (split_x, 0), (split_x, combined.shape[0]), (255, 255, 255), 2)
    cv2.putText(combined, "Expert", (18, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Learner", (split_x + 18, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return ensure_even_frame_size(combined)


def resize_to_height(frame: Any, target_height: int) -> Any:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def ensure_even_frame_size(frame: Any) -> Any:
    height, width = frame.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height == height and even_width == width:
        return frame
    return frame[:even_height, :even_width]


def dtw_window_ratio(expert_angles: list[float], learner_angles: list[float]) -> float:
    max_length = max(len(expert_angles), len(learner_angles), 1)
    required_window_ratio = (abs(len(expert_angles) - len(learner_angles)) + 1) / max_length
    return min(1.0, max(0.1, required_window_ratio + 0.05))


def angle_difference_degrees(angle_a: float, angle_b: float) -> float:
    angle_diff = abs(angle_a - angle_b)
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    return abs(angle_diff)


def draw_learner_compare_text(
    frame: Any,
    *,
    learner_angle: float,
    expert_reference_angle: float,
    angle_difference: float,
) -> None:
    lines = [
        f"Learner angle: {learner_angle:.1f} deg",
        f"Expert ref angle: {expert_reference_angle:.1f} deg",
        f"Difference: {angle_difference:.1f} deg",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28


def draw_angle_difference_arc(
    frame: Any,
    *,
    center: tuple[float, float],
    learner_angle: float,
    expert_reference_angle: float,
    angle_difference: float,
) -> None:
    center_point = (int(round(center[0])), int(round(center[1])))
    text_position = (center_point[0] + 70, max(24, center_point[1] - 10))

    if angle_difference > 3:
        visual_diff = draw_small_angle_arc(
            frame,
            center=center_point,
            expert_angle=expert_reference_angle,
            learner_angle=learner_angle,
        )
        cv2.putText(
            frame,
            f"Angle diff: {visual_diff:.1f} deg",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            f"Aligned: {angle_difference:.1f} deg",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_small_angle_arc(
    frame: Any,
    center: tuple[int, int],
    expert_angle: float,
    learner_angle: float,
    radius: int = 55,
    steps: int = 25,
) -> float:
    a1 = expert_angle % 180
    a2 = learner_angle % 180

    diff = ((a2 - a1 + 90) % 180) - 90
    start = a1
    points = make_arc_points(center, start, diff, radius=radius, steps=steps)

    avg_y = sum(point[1] for point in points) / len(points)
    if avg_y > center[1]:
        points = make_arc_points(center, start + 180, diff, radius=radius, steps=steps)

    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(frame, p1, p2, (0, 255, 255), 4)

    cv2.line(frame, center, points[0], (0, 255, 255), 2)
    cv2.line(frame, center, points[-1], (0, 255, 255), 2)

    return abs(diff)


def make_arc_points(
    center: tuple[int, int],
    start_angle: float,
    diff: float,
    radius: int = 60,
    steps: int = 30,
) -> list[tuple[int, int]]:
    points = []
    for k in range(steps + 1):
        t = k / steps
        angle_radians = math.radians(start_angle + diff * t)
        x = int(center[0] + radius * math.cos(angle_radians))
        y = int(center[1] + radius * math.sin(angle_radians))
        points.append((x, y))
    return points


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

