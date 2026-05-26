import cv2
import numpy as np
from ultralytics import YOLO

from core.display_mode import DEFAULT_DISPLAY_MODE, DisplayMode
from core.settings import TRACKER
from core.renderer import render_dot_mask


class VideoPipeline:
    def __init__(self, weights: str) -> None:
        self.model = YOLO(weights)
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True


    def run(
        self,
        video_path: str,
        frame_callback=None,
        output_path: str | None = None,
        display_mode_fn=None,
        show_mask_fn=None,
    ) -> None:
        self._stop_flag = False

        cap = self._open_capture(video_path)
        writer = self._open_writer(output_path, cap)

        try:
            self._process_loop(cap, writer, frame_callback, display_mode_fn, show_mask_fn)
        finally:
            self._release(cap, writer)


    @staticmethod
    def _open_capture(video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        return cap

    @staticmethod
    def _open_writer(
        output_path: str | None,
        cap: cv2.VideoCapture,
    ) -> cv2.VideoWriter | None:
        if not output_path:
            return None
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    def _process_loop(self, cap, writer, frame_callback, display_mode_fn, show_mask_fn) -> None:
        while cap.isOpened() and not self._stop_flag:
            ok, frame = cap.read()
            if not ok:
                break
            rendered = self._render_frame(frame, display_mode_fn, show_mask_fn)
            if writer:
                writer.write(rendered)
            if frame_callback:
                frame_callback(rendered)

    def _render_frame(self, frame: np.ndarray, display_mode_fn=None, show_mask_fn=None) -> np.ndarray:
        mode = self._resolve_display_mode(display_mode_fn, show_mask_fn)
        if mode == DisplayMode.ORIGINAL:
            return frame

        h, w = frame.shape[:2]
        results = self.model.track(frame, **TRACKER)

        if mode == DisplayMode.DEFAULT:
            return results[0].plot()

        if mode == DisplayMode.ALL:
            # TODO: реализовать полноценный режим "Всё" позже. Сейчас это только заглушка.
            return render_dot_mask(results, w, h, frame)

        return render_dot_mask(results, w, h, frame)

    @staticmethod
    def _resolve_display_mode(display_mode_fn=None, show_mask_fn=None) -> DisplayMode:
        if display_mode_fn is not None:
            return display_mode_fn()
        if show_mask_fn is None:
            return DEFAULT_DISPLAY_MODE
        return DisplayMode.CORTICAL_VISION if show_mask_fn() else DisplayMode.DEFAULT

    @staticmethod
    def _release(cap: cv2.VideoCapture, writer: cv2.VideoWriter | None) -> None:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
