import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QCheckBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from core.settings import WEIGHTS_PATH , SAMPLES_DIR
from core.pipeline import VideoPipeline


class WorkerThread(QThread):
    finished   = pyqtSignal()
    frame_ready = pyqtSignal(object)

    def __init__(self, pipeline: VideoPipeline, video_path: str, show_mask_fn):
        super().__init__()
        self.pipeline   = pipeline
        self.video_path = video_path
        self.show_mask_fn = show_mask_fn

    def run(self):
        self.pipeline.run(self.video_path, frame_callback=self.frame_ready.emit, show_mask_fn = self.show_mask_fn)
        self.finished.emit()
        


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("cv")
        self.resize(1200, 800)
        self.pipeline = VideoPipeline(WEIGHTS_PATH )
        self.worker = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.combo = QComboBox()
        files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".mp4", ".jpg")))
        self.combo.addItems(files)
        layout.addWidget(self.combo)

        self.btn = QPushButton("Запустити")
        self.btn.clicked.connect(self.start_tracking)
        layout.addWidget(self.btn)

        self.mask_checkbox = QCheckBox("Кортикальний зор")
        self.mask_checkbox.setChecked(True)
        layout.addWidget(self.mask_checkbox)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.video_label)



    def start_tracking(self):
        filename = self.combo.currentText()
        filepath = os.path.join(SAMPLES_DIR, filename)

        self.btn.setEnabled(False)
        self.worker = WorkerThread(self.pipeline, filepath, show_mask_fn=self.mask_checkbox.isChecked)
        self.worker.finished.connect(lambda: self.btn.setEnabled(True))
        self.worker.frame_ready.connect(self.update_frame)  # ← підключаємо сигнал
        self.worker.start()

    def update_frame(self, frame):
        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(img)
        

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)