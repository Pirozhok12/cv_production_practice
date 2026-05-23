from pathlib import Path


ROOT = Path(__file__).parent.parent  

WEIGHTS_PATH  = 'data/runs/segment/train_yolov8m-seg_Face_Body_8epochs/weights/best.pt'
SAMPLES_DIR = 'data/samples'
OUTPUT_DIR   = ROOT / "outputs"

CANNY_LOW = 30          # нижний порог чувствительности краёв
CANNY_HIGH = 90         # верхний порог
CANNY_GRID = 16         # плотность сетки (меньше = плотнее)
CANNY_MIN_STRENGTH = 10 # минимальная сила края чтобы рисовать точку

TRACKER = {
    "persist": True,
    "iou": 0.4,
    "conf": 0.25,
    "tracker": "bytetrack.yaml",
    "imgsz": 640,
    "verbose": False
}