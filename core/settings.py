from pathlib import Path


ROOT = Path(__file__).parent.parent  

WEIGHTS_PATH  = 'data/runs/segment/train_yolov8m-seg_Face_Body_8epochs/weights/best.pt'
SAMPLES_DIR = 'data/samples'
OUTPUT_DIR   = ROOT / "outputs"

CANNY_LOW = 60          # нижний порог чувствительности краёв
CANNY_HIGH = 150         # верхний порог
CANNY_GRID = 8          # плотность сетки (меньше = плотнее)
CANNY_MIN_STRENGTH = 10 # минимальная сила края чтобы рисовать точку

TRACKER = {
    "persist": True,
    "iou": 0.35,
    "conf": 0.25,
    "tracker": "bytetrack.yaml",
    "imgsz": 640,
    "verbose": False
}   


PERSON_CLASS_NAMES = "person_poly"

FACE_PART_CLASS_NAMES = {
    "skin", "l_brow", "r_brow", "l_eye", "r_eye",
    "nose", "upper_lip", "mouth", "lower_lip", "hair"
}

PERSON_CONF_THRESHOLD = 0.25
MASK_THRESHOLD = 0.3
AREA_WEIGHT = 0.80
CENTER_WEIGHT = 0.10
CONF_WEIGHT = 0.10
RELATED_MASK_MIN_OVERLAP = 0.05