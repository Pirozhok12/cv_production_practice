import os
import shutil

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QComboBox

from ui.components import FixedHeightItemDelegate
from ui.styles import CONTROL_COMBO_STYLE, CONTROL_HEIGHT, FILE_MENU_ITEM_HEIGHT, PREVIEW_SIZE


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SAMPLE_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
CHOOSE_OTHER_FILE_VALUE = "__choose_other_file__"
CHOOSE_OTHER_FILE_LABEL = "Обрати інше зображення або відео..."


def create_file_combo(samples_dir: str) -> QComboBox:
    combo = QComboBox()
    combo.setMinimumHeight(CONTROL_HEIGHT)
    combo.setFixedHeight(CONTROL_HEIGHT)
    combo.setIconSize(QSize(PREVIEW_SIZE, PREVIEW_SIZE))
    combo.setItemDelegate(FixedHeightItemDelegate(FILE_MENU_ITEM_HEIGHT, combo))
    combo.setStyleSheet(CONTROL_COMBO_STYLE)
    populate_file_combo(combo, samples_dir)
    combo.setCurrentIndex(-1)
    return combo


def populate_file_combo(combo: QComboBox, samples_dir: str) -> None:
    files = []
    if os.path.isdir(samples_dir):
        files = sorted(
            f for f in os.listdir(samples_dir)
            if f.lower().endswith(SAMPLE_EXTENSIONS)
        )
    for filename in files:
        combo.addItem(create_file_list_item_icon(samples_dir, filename), filename, filename)
    combo.addItem(CHOOSE_OTHER_FILE_LABEL, CHOOSE_OTHER_FILE_VALUE)


def create_file_list_item_icon(samples_dir: str, filename: str) -> QIcon:
    filepath = os.path.join(samples_dir, filename)
    return create_sample_preview_icon(filepath)


def refresh_file_combo(combo: QComboBox, samples_dir: str, selected_filename: str | None = None) -> None:
    combo.blockSignals(True)
    combo.clear()
    populate_file_combo(combo, samples_dir)
    if selected_filename is None:
        combo.setCurrentIndex(-1)
    else:
        index = combo.findData(selected_filename)
        combo.setCurrentIndex(index)
    combo.blockSignals(False)


def is_choose_other_file_item(value) -> bool:
    return value == CHOOSE_OTHER_FILE_VALUE


def is_supported_sample_file(filename: str) -> bool:
    return filename.lower().endswith(SAMPLE_EXTENSIONS)


def select_first_supported_file(paths: list[str]) -> str | None:
    for path in paths:
        if is_supported_sample_file(path):
            return path
    return None


def build_unique_sample_filename(source_path: str, samples_dir: str) -> str:
    original_name = os.path.basename(source_path)
    stem, extension = os.path.splitext(original_name)
    candidate = original_name
    index = 1

    while os.path.exists(os.path.join(samples_dir, candidate)):
        candidate = f"{stem}_{index}{extension}"
        index += 1

    return candidate


def copy_sample_file(source_path: str, samples_dir: str) -> str:
    filename = build_unique_sample_filename(source_path, samples_dir)
    destination_path = os.path.join(samples_dir, filename)
    shutil.copy2(source_path, destination_path)
    return filename


def create_sample_preview_icon(filepath: str) -> QIcon:
    frame = _read_preview_frame(filepath)
    if frame is None:
        return QIcon()

    pixmap = _pixmap_from_bgr_frame(frame)
    scaled = pixmap.scaled(
        QSize(PREVIEW_SIZE, PREVIEW_SIZE),
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    return QIcon(_crop_center(scaled, PREVIEW_SIZE))


def is_video_file(filename: str) -> bool:
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def _read_preview_frame(filepath: str):
    if filepath.lower().endswith(IMAGE_EXTENSIONS):
        encoded = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if filepath.lower().endswith(VIDEO_EXTENSIONS):
        cap = cv2.VideoCapture(filepath)
        try:
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()
    return None


def _pixmap_from_bgr_frame(frame) -> QPixmap:
    h, w = frame.shape[:2]
    image = QImage(frame.data, w, h, w * 3, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(image)


def _crop_center(pixmap: QPixmap, size: int) -> QPixmap:
    x = max(0, (pixmap.width() - size) // 2)
    y = max(0, (pixmap.height() - size) // 2)
    return pixmap.copy(x, y, size, size)
