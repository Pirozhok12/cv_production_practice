from pathlib import Path


ROOT = Path(__file__).parent.parent  

WEIGHTS_PATH  = 'data/runs/segment/train_yolo11m-seg/weights/best.pt'
SAMPLES_DIR = 'data/samples'
OUTPUT_DIR   = ROOT / "outputs"


TRACKER = {
    "persist": True,
    "iou": 0.65,
    "conf": 0.85,
    "tracker": "bytetrack.yaml",
    "imgsz": 640,
    "verbose": False
}

DOT_SPACING = 6
DOT_RADIUS  = 1