import cv2
import numpy as np
from ultralytics import YOLO

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
        show_mask_fn=None,
    ) -> None:
        self._stop_flag = False

        cap = self._open_capture(video_path)
        writer = self._open_writer(output_path, cap)

        try:
            self._process_loop(cap, writer, frame_callback, show_mask_fn)
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

    def _process_loop(self, cap, writer, frame_callback, show_mask_fn) -> None:
        while cap.isOpened() and not self._stop_flag:
            ok, frame = cap.read()
            if not ok:
                break
            rendered = self._render_frame(frame, show_mask_fn)
            if writer:
                writer.write(rendered)
            if frame_callback:
                frame_callback(rendered)

    def _render_frame(self, frame: np.ndarray, show_mask_fn) -> np.ndarray:
        h, w = frame.shape[:2]
        results = self.model.track(frame, **TRACKER)
        if show_mask_fn is None or show_mask_fn():
            return render_dot_mask(results, w, h, frame)
        return results[0].plot()

    @staticmethod
    def _release(cap: cv2.VideoCapture, writer: cv2.VideoWriter | None) -> None:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()