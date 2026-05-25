import os
import time
from typing import Callable

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QCheckBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent

from core.settings import WEIGHTS_PATH, SAMPLES_DIR
from core.pipeline import VideoPipeline


class WorkerThread(QThread):
    frame_ready = pyqtSignal(object) 

    def __init__(self, pipeline: VideoPipeline, video_path: str, show_mask_fn: Callable[[], bool]):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = video_path
        self.show_mask_fn = show_mask_fn

    def run(self) -> None:
        self.pipeline.run(
            self.video_path,
            frame_callback=self.frame_ready.emit,
            show_mask_fn=self.show_mask_fn,
        )
        self.finished.emit()  


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("cv")
        self.resize(1200, 800)
        self.setMinimumSize(400, 300)

        self.pipeline = VideoPipeline(WEIGHTS_PATH)
        self.worker: WorkerThread | None = None
        self._last_frame_time: float | None = None

        self._build_ui()
        self._on_file_changed(self.combo.currentText())



    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setLayout(root)

        # Video area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black;")
        root.addWidget(self.video_label, stretch=1)

        # FPS overlay (child of video_label so it floats over it)
        self.fps_label = QLabel("FPS: —", parent=self.video_label)
        self.fps_label.setVisible(False)
        self.fps_label.setStyleSheet(
            "color: #00ff88; background: rgba(0,0,0,160);"
            "padding: 2px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.fps_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Bottom bar
        bottom = QWidget()
        bottom.setMaximumHeight(64)
        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 8)
        bar.setSpacing(10)
        bottom.setLayout(bar)

        self.combo = QComboBox()
        self._populate_combo()
        self.combo.currentTextChanged.connect(self._on_file_changed)
        bar.addWidget(self.combo, stretch=1)

        self.mask_checkbox = QCheckBox("Кортикальний зор")
        self.mask_checkbox.setChecked(True)
        bar.addWidget(self.mask_checkbox)

        self.btn_start = QPushButton("Запустити")
        self.btn_start.setFixedWidth(110)
        self.btn_start.clicked.connect(self.start_tracking)
        bar.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.setFixedWidth(80)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.stop_tracking)
        bar.addWidget(self.btn_stop)

        root.addWidget(bottom)

    def resizeEvent(self, a0: QResizeEvent | None) -> None: 
        super().resizeEvent(a0)
        self._reposition_fps()

    def _reposition_fps(self) -> None:
        self.fps_label.adjustSize()
        x = self.video_label.width() - self.fps_label.width() - 10
        self.fps_label.move(x, 10)

    def _populate_combo(self) -> None:
        files = sorted(
            f for f in os.listdir(SAMPLES_DIR)
            if f.lower().endswith((".mp4", ".jpg"))
        )
        self.combo.addItems(files)

    @staticmethod
    def _is_video(filename: str) -> bool:
        return filename.lower().endswith(".mp4")



    def _on_file_changed(self, filename: str) -> None:
        self.btn_stop.setVisible(self._is_video(filename))
        self.fps_label.setVisible(False)

    def start_tracking(self) -> None:
        filename = self.combo.currentText()
        filepath = os.path.join(SAMPLES_DIR, filename)
        is_video = self._is_video(filename)

        self._last_frame_time = None
        self.btn_start.setEnabled(False)
        self.combo.setEnabled(False)

        if is_video:
            self.btn_stop.setEnabled(True)
            self.fps_label.setText("FPS: —")
            self.fps_label.setVisible(True)
            self._reposition_fps()

        self.worker = WorkerThread(
            self.pipeline,
            filepath,
            show_mask_fn=self.mask_checkbox.isChecked,
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.start()

    def stop_tracking(self) -> None:
        if hasattr(self.pipeline, "stop"):
            self.pipeline.stop()
        self.btn_stop.setEnabled(False)

    def _on_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo.setEnabled(True)
        self.fps_label.setVisible(False)



    def update_frame(self, frame) -> None:
        now = time.perf_counter()
        if self._last_frame_time is not None:
            fps = 1.0 / (now - self._last_frame_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._reposition_fps()
        self._last_frame_time = now

        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(img)

        target = self.video_label.contentsRect().size()
        if target.isEmpty():
            target = self.video_label.size()

        scaled = pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)