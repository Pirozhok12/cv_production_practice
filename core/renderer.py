import cv2
import numpy as np
from core.settings import CANNY_LOW, CANNY_HIGH, CANNY_GRID, CANNY_MIN_STRENGTH
from core.selector import collect_related_masks, select_main_person


def _get_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def _dot_color_and_radius(strength, grid):
    t = strength / 255.0
    color = (int(255 * t), int(180 * (1 - t)), int(255 * (1 - t)))
    radius = max(2, int(grid * 0.15 + t * grid * 0.2))
    return color, radius


def _process_cell(canvas, edges, binary, gx, gy):
    cell_mask = binary[gy:gy+CANNY_GRID, gx:gx+CANNY_GRID]
    cell_edges = edges[gy:gy+CANNY_GRID, gx:gx+CANNY_GRID]

    if not cell_mask.any():
        return

    strength = cell_edges[cell_mask].mean()
    if strength < CANNY_MIN_STRENGTH:
        return

    edge_points = np.argwhere((cell_edges * cell_mask) > 0)
    if len(edge_points) == 0:
        return

    cy_local, cx_local = edge_points[len(edge_points) // 2]
    color, radius = _dot_color_and_radius(strength, CANNY_GRID)
    cv2.circle(canvas, (gx + cx_local, gy + cy_local), radius, color, -1)


def render_dot_masks(binary_masks, frame_width, frame_height, frame):
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    if not binary_masks:
        return canvas

    edges = _get_edges(frame)

    for binary in binary_masks:
        if binary.shape != (frame_height, frame_width):
            binary = cv2.resize(
                binary.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        for gy in range(0, frame_height, CANNY_GRID):
            for gx in range(0, frame_width, CANNY_GRID):
                _process_cell(canvas, edges, binary, gx, gy)

    return canvas


def render_dot_mask(results, frame_width, frame_height, frame):
    if not results:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    result = results[0]
    if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    selected_person = select_main_person(result, frame_width, frame_height)
    related_masks = collect_related_masks(result, selected_person, frame_width, frame_height)
    return render_dot_masks(related_masks, frame_width, frame_height, frame)