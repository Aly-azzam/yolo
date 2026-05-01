Place your Roboflow-exported Ultralytics weights here as:

  scissors_yolo.pt

The app returns HTTP 503 with a clear message if this file is missing.

Alternative: set environment variable SCISSORS_YOLO_PT to the full path of any
compatible .pt file (e.g. your Roboflow download or runs/detect/train/weights/best.pt).
