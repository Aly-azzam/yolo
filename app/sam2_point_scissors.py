"""Compatibility wrapper for the SAM2+YOLO scissors tracker.

Prefer running:
python -m app.services.sam2.sam2_yolo_tracker --video_path ... --yolo_model ... --stride 5 --use_gpu
"""

from app.services.sam2.sam2_yolo_tracker import main, track_scissors_with_sam2

__all__ = ["main", "track_scissors_with_sam2"]


if __name__ == "__main__":
    main()
