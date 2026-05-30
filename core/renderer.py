import cv2
import numpy as np
from core.settings import CANNY_LOW, CANNY_HIGH, CANNY_GRID, CANNY_MIN_STRENGTH
from core.selector import collect_related_masks, select_main_person


def _get_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # bilateralFilter заменён на GaussianBlur — в 10x быстрее
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def render_dot_masks(binary_masks, frame_width, frame_height, frame):
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    if not binary_masks:
        return canvas

    edges = _get_edges(frame)
    g = CANNY_GRID

    # Объединяем все маски в одну
    combined = np.zeros((frame_height, frame_width), dtype=bool)
    for binary in binary_masks:
        if binary.shape != (frame_height, frame_width):
            binary = cv2.resize(
                binary.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        combined |= binary

    # Маскируем края
    masked_edges = edges * combined

    # Считаем среднюю силу края по клеткам векторно
    h_cells = frame_height // g
    w_cells = frame_width // g

    # Обрезаем до кратного размера
    crop_h, crop_w = h_cells * g, w_cells * g
    me = masked_edges[:crop_h, :crop_w].reshape(h_cells, g, w_cells, g)
    strength_grid = me.mean(axis=(1, 3))  # (h_cells, w_cells)

    # Находим клетки где сила края достаточна
    ys, xs = np.nonzero(strength_grid >= CANNY_MIN_STRENGTH)

    for cy, cx in zip(ys, xs):
        gy, gx = cy * g, cx * g
        cell = masked_edges[gy:gy+g, gx:gx+g]
        pts = np.argwhere(cell > 0)
        if len(pts) == 0:
            continue
        ly, lx = pts[len(pts) // 2]
        strength = float(strength_grid[cy, cx])
        t = min(strength / 255.0, 1.0)
        color = (int(255 * t), int(180 * (1 - t)), int(255 * (1 - t)))
        radius = max(2, int(g * 0.15 + t * g * 0.2))
        cv2.circle(canvas, (gx + lx, gy + ly), radius, color, -1)

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


def render_dot_frame(frame_width, frame_height, frame):
    full_mask = np.ones((frame_height, frame_width), dtype=bool)
    return render_dot_masks([full_mask], frame_width, frame_height, frame)