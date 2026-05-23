import cv2
from ultralytics import YOLO
from core.settings import TRACKER
from core.renderer import render_dot_mask
import numpy as np

class VideoPipeline:
    def __init__(self, weights: str):
        self.model = YOLO(weights)

    def run(self, video_path: str, frame_callback=None, output_path: str | None = None, show_mask_fn=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                results = self.model.track(frame, **TRACKER)
                
                if show_mask_fn is None or show_mask_fn():
                    rendered = render_dot_mask(results, width, height, frame)
                else:
                    rendered = rendered = results[0].plot()              

                if writer:
                    writer.write(rendered)
                if frame_callback:
                    frame_callback(rendered)
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()