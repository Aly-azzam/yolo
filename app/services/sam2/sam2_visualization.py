from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.sam2.sam2_schema import Sam2FrameResult


def write_mask_overlay_video(
    *,
    frame_dir: Path,
    frame_results: list[Sam2FrameResult],
    masks_by_processed_frame: dict[int, np.ndarray],
    output_path: Path,
    fps: float,
    trajectory_metrics: dict | None = None,
) -> None:
    first_frame = cv2.imread(str(frame_dir / "000000.jpg"))
    if first_frame is None:
        raise RuntimeError(f"Could not read extracted frame from {frame_dir}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create visualization video at {output_path}")

    trajectory_by_processed_frame = _trajectory_points_by_processed_frame(frame_results)
    fitted_line = (trajectory_metrics or {}).get("fitted_line")

    try:
        drawn_points: list[tuple[int, int]] = []
        for result in frame_results:
            processed_idx = result.processed_frame_index
            if processed_idx is None:
                continue

            frame_path = frame_dir / f"{processed_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            mask = masks_by_processed_frame.get(processed_idx)
            if mask is not None:
                frame = _overlay_mask(frame, mask)

            if result.bbox is not None:
                x1, y1, x2, y2 = result.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if result.centroid is not None:
                cx, cy = map(int, result.centroid)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            trajectory_point = trajectory_by_processed_frame.get(processed_idx)
            if trajectory_point is not None:
                drawn_points.append(trajectory_point)
                _draw_trajectory(frame, drawn_points, fitted_line)
                _draw_deviation(frame, trajectory_point, fitted_line)

            cv2.putText(
                frame,
                f"SAM2 scissors frame={result.frame_index} point=bbox_center",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)


def _overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != frame.shape[:2]:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    color = np.zeros_like(frame)
    color[:, :] = (30, 180, 30)
    blended = cv2.addWeighted(frame, 0.65, color, 0.35, 0)
    frame[mask_bool] = blended[mask_bool]
    return frame


def _trajectory_points_by_processed_frame(
    frame_results: list[Sam2FrameResult],
) -> dict[int, tuple[int, int]]:
    points: dict[int, tuple[int, int]] = {}
    for result in frame_results:
        if result.processed_frame_index is None or result.trajectory_point is None:
            continue
        x, y = result.trajectory_point
        points[result.processed_frame_index] = (int(round(x)), int(round(y)))
    return points


def _draw_trajectory(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    fitted_line: dict | None,
) -> None:
    if len(points) >= 2:
        cv2.polylines(
            frame,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=(255, 0, 255),
            thickness=2,
        )

    for point in points[-30:]:
        cv2.circle(frame, point, 4, (255, 0, 255), -1)

    current = points[-1]
    cv2.circle(frame, current, 7, (255, 255, 0), -1)
    cv2.putText(
        frame,
        "trajectory point",
        (current[0] + 8, max(24, current[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if fitted_line:
        start = tuple(int(round(value)) for value in fitted_line["start_point"])
        end = tuple(int(round(value)) for value in fitted_line["end_point"])
        cv2.line(frame, start, end, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "fitted straight line",
            (start[0], max(24, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_deviation(
    frame: np.ndarray,
    point: tuple[int, int],
    fitted_line: dict | None,
) -> None:
    if not fitted_line:
        return

    line_point = np.array(fitted_line["point"], dtype=np.float32)
    direction = np.array(fitted_line["direction"], dtype=np.float32)
    current = np.array(point, dtype=np.float32)
    projection = line_point + direction * float((current - line_point) @ direction)
    projection_point = tuple(int(round(value)) for value in projection.tolist())
    cv2.line(frame, point, projection_point, (0, 165, 255), 1, cv2.LINE_AA)
