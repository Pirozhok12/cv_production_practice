import os
import time
from typing import Callable

from core.display_mode import DisplayMode
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from core.pipeline import VideoPipeline
from core.settings import SAMPLES_DIR, WEIGHTS_PATH
from ui.components import FileDropArea, create_fps_label, create_video_label
from ui.controls import (
    create_control_panel,
    set_stop_button_state,
    set_tracking_controls_enabled,
)
from ui.file_list import (
    SAMPLE_EXTENSIONS,
    copy_sample_file,
    is_image_file,
    is_choose_other_file_item,
    is_supported_sample_file,
    is_video_file,
    refresh_file_combo,
    select_first_supported_file,
)
from ui.styles import FPS_MARGIN, ROOT_MARGINS, ROOT_SPACING, WINDOW_MIN_SIZE, WINDOW_SIZE, WINDOW_TITLE


class WorkerThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, pipeline: VideoPipeline, video_path: str, display_mode_fn: Callable[[], DisplayMode]):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = video_path
        self.display_mode_fn = display_mode_fn

    def run(self) -> None:
        self.pipeline.run(
            self.video_path,
            frame_callback=self.frame_ready.emit,
            display_mode_fn=self.display_mode_fn,
        )
        self.finished.emit()


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*WINDOW_SIZE)
        self.setMinimumSize(*WINDOW_MIN_SIZE)

        self.pipeline = VideoPipeline(WEIGHTS_PATH)
        self.worker: WorkerThread | None = None
        self._last_frame_time: float | None = None

        self._build_ui()
        self._on_file_changed(self.combo.currentText())

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(*ROOT_MARGINS)
        root.setSpacing(ROOT_SPACING)
        self.setLayout(root)

        self.preview_stack = QStackedWidget()
        root.addWidget(self.preview_stack, stretch=1)

        self.drop_area = FileDropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        self.drop_area.select_requested.connect(self.choose_input_file)
        self.preview_stack.addWidget(self.drop_area)

        self.video_label = create_video_label()
        self.preview_stack.addWidget(self.video_label)

        self.fps_label = create_fps_label(self.video_label)

        self.controls = create_control_panel()
        self.combo = self.controls.combo
        self.mode_dropdown = self.controls.mode_dropdown
        self.btn_start = self.controls.btn_start
        self.btn_stop = self.controls.btn_stop

        self._previous_file_index = -1
        self.combo.currentIndexChanged.connect(self._on_file_index_changed)
        self.mode_dropdown.currentIndexChanged.connect(self._on_display_mode_changed)
        self.btn_start.clicked.connect(self.start_tracking)
        self.btn_stop.clicked.connect(self.stop_tracking)

        self.display_mode = self._current_display_mode()

        root.addWidget(self.controls.widget)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._reposition_fps()

    def _reposition_fps(self) -> None:
        self.fps_label.adjustSize()
        x = self.video_label.width() - self.fps_label.width() - FPS_MARGIN
        self.fps_label.move(x, FPS_MARGIN)

    @staticmethod
    def _is_video(filename: str) -> bool:
        return is_video_file(filename)

    @staticmethod
    def _should_show_stop(filename: str) -> bool:
        if is_image_file(filename):
            return False
        return True

    def _current_display_mode(self) -> DisplayMode:
        return self.mode_dropdown.currentValue()

    def _on_display_mode_changed(self, *_args) -> None:
        self.display_mode = self._current_display_mode()

    def _on_file_index_changed(self, index: int) -> None:
        value = self.combo.itemData(index)
        if is_choose_other_file_item(value):
            if not self.choose_input_file():
                self._restore_previous_file_selection()
            return

        if index >= 0:
            self._previous_file_index = index
        self._on_file_changed(value if isinstance(value, str) else "")

    def _on_file_changed(self, filename: str) -> None:
        has_file = bool(filename)
        self.preview_stack.setCurrentWidget(self.video_label if has_file else self.drop_area)
        self.btn_start.setEnabled(has_file)
        set_stop_button_state(
            self.controls,
            self.btn_stop.isEnabled() and has_file,
            self._should_show_stop(filename) if has_file else False,
        )
        self.fps_label.setVisible(False)

    def _on_files_dropped(self, paths: list[str]) -> None:
        source_path = select_first_supported_file(paths)
        if source_path is None:
            self._show_input_error("Формат файлу не підтримується")
            return
        self.add_input_file(source_path)

    def choose_input_file(self) -> bool:
        filters = "Зображення і відео (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm)"
        source_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Оберіть зображення або відео",
            "",
            filters,
        )
        if not source_path:
            return False
        return self.add_input_file(source_path)

    def add_input_file(self, source_path: str) -> bool:
        if not source_path or not os.path.isfile(source_path):
            self._show_input_error("Не вдалося відкрити файл")
            return False

        if not is_supported_sample_file(source_path):
            self._show_input_error(
                "Формат файлу не підтримується. Доступні формати: "
                + ", ".join(SAMPLE_EXTENSIONS)
            )
            return False

        try:
            os.makedirs(SAMPLES_DIR, exist_ok=True)
        except OSError as exc:
            self._show_input_error(f"Не вдалося створити папку для файлів: {exc}")
            return False

        try:
            filename = copy_sample_file(source_path, SAMPLES_DIR)
        except OSError as exc:
            self._show_input_error(f"Не вдалося скопіювати файл: {exc}")
            return False

        refresh_file_combo(self.combo, SAMPLES_DIR, filename)
        self._previous_file_index = self.combo.currentIndex()
        self._on_file_changed(filename)
        return True

    def _show_input_error(self, message: str) -> None:
        QMessageBox.warning(self, "Не вдалося додати файл", message)

    def _restore_previous_file_selection(self) -> None:
        self.combo.blockSignals(True)
        if self._previous_file_index >= 0 and self._previous_file_index < self.combo.count():
            self.combo.setCurrentIndex(self._previous_file_index)
            value = self.combo.itemData(self._previous_file_index)
            filename = value if isinstance(value, str) and not is_choose_other_file_item(value) else ""
        else:
            self.combo.setCurrentIndex(-1)
            filename = ""
        self.combo.blockSignals(False)
        self._on_file_changed(filename)

    def start_tracking(self) -> None:
        value = self.combo.currentData()
        filename = value if isinstance(value, str) and not is_choose_other_file_item(value) else ""
        if not filename:
            self._show_input_error("Не вдалося відкрити файл")
            return

        filepath = os.path.join(SAMPLES_DIR, filename)
        is_video = self._is_video(filename)

        self._last_frame_time = None
        set_tracking_controls_enabled(self.controls, False)

        if is_video:
            set_stop_button_state(self.controls, True)
            self.fps_label.setText("FPS: —")
            self.fps_label.setVisible(True)
            self._reposition_fps()

        self.worker = WorkerThread(
            self.pipeline,
            filepath,
            display_mode_fn=lambda: self.display_mode,
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.start()

    def stop_tracking(self) -> None:
        if hasattr(self.pipeline, "stop"):
            self.pipeline.stop()
        set_stop_button_state(self.controls, False)

    def _on_finished(self) -> None:
        set_tracking_controls_enabled(self.controls, True)
        set_stop_button_state(self.controls, False)
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
