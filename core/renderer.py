import cv2
import numpy as np
from core.settings import DOT_RADIUS, DOT_SPACING

def render_dot_mask( results, frame_width, frame_height):

    canvas = np.zeros((frame_height, frame_width), dtype=np.float32)
    dot_spacing = DOT_SPACING
    max_radius = DOT_RADIUS 
    if results[0].masks is not None:

        for mask in results[0].masks.data.cpu().numpy():

            # resize soft mask
            soft_mask = cv2.resize(
                mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR
            )

            for y in range(0, frame_height, dot_spacing):
                for x in range(0, frame_width, dot_spacing):

                    value = soft_mask[y, x]  

                    if value > 0.01:

                        intensity = int(value * 255)


                        radius = max(1, int(value * max_radius))

                        cv2.circle(
                            canvas,
                            (x, y),
                            radius,
                            intensity,
                            -1
                        )

    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)