from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

from core.settings import (
    AREA_WEIGHT,
    CENTER_WEIGHT,
    CONF_WEIGHT,
    FACE_PART_CLASS_NAMES,
    MASK_THRESHOLD,
    PERSON_CLASS_NAMES,
    PERSON_CONF_THRESHOLD,
    RELATED_MASK_MIN_OVERLAP,
)

BBox = tuple[float, float, float, float]


@dataclass
class SelectedPerson:
    index: int
    bbox_xyxy: BBox
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray
    mask_area: int
    center_bias: float
    score: float
    track_id: int | None = None
    candidate_type: str = "person"
    related_indices: tuple[int, ...] = ()


# ------------------------------------------------------------------ tensor helpers

def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value, default: int | float = 0) -> int | float:
    array = _to_numpy(value)
    if array is None or array.size == 0:
        return default
    return array.reshape(-1)[0].item()


def _get_row(values, index, default=None):
    array = _to_numpy(values)
    if array is None or index >= len(array):
        return default
    return array[index]


def _box_count(boxes) -> int:
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return 0
    xyxy = _to_numpy(boxes.xyxy)
    return 0 if xyxy is None else len(xyxy)


def _mask_count(masks) -> int:
    if masks is None or getattr(masks, "data", None) is None:
        return 0
    return len(masks.data)


def _value_count(value) -> int:
    array = _to_numpy(value)
    return 0 if array is None else len(array)


def _track_id_at(boxes, index) -> int | None:
    ids = getattr(boxes, "id", None)
    if ids is None:
        return None
    value = _get_row(ids, index)
    return None if value is None else int(_scalar(value))


# ------------------------------------------------------------------ geometry

def _bbox_from_row(row) -> BBox:
    x1, y1, x2, y2 = row
    return (float(x1), float(y1), float(x2), float(y2))


def _bbox_union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, int((x2 - x1) * (y2 - y1)))


def _bboxes_overlap_or_close(a: BBox, b: BBox) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2:
        return True
    gap_x = max(bx1 - ax2, ax1 - bx2, 0)
    gap_y = max(by1 - ay2, ay1 - by2, 0)
    gap = hypot(gap_x, gap_y)
    largest_side = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1)
    return gap <= max(24.0, largest_side * 0.35)


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_center_bias(center: tuple[float, float], frame_width: int, frame_height: int) -> float:
    frame_center = (frame_width / 2.0, frame_height / 2.0)
    distance = hypot(center[0] - frame_center[0], center[1] - frame_center[1])
    max_distance = hypot(frame_center[0], frame_center[1])
    if max_distance <= 0:
        return 1.0
    return 1.0 - min(distance / max_distance, 1.0)


# ------------------------------------------------------------------ mask helpers

def get_class_name(result, class_id) -> str:
    names = getattr(result, "names", None)
    class_id = int(class_id)
    if isinstance(names, dict):
        return str(names.get(class_id, "")).lower()
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id]).lower()
    return ""


def resize_mask_to_frame(
    mask_tensor, frame_width: int, frame_height: int, threshold: float = MASK_THRESHOLD
) -> np.ndarray:
    mask = _to_numpy(mask_tensor)
    if mask is None or mask.size == 0:
        return np.zeros((frame_height, frame_width), dtype=bool)
    mask = np.squeeze(mask).astype(np.float32, copy=False)
    if mask.shape != (frame_height, frame_width):
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    return mask > threshold


def compute_mask_area(binary_mask: np.ndarray) -> int:
    return int(np.count_nonzero(binary_mask))


def _mask_center(binary_mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(binary_mask)
    if len(points) == 0:
        return None
    y, x = points.mean(axis=0)
    return (float(x), float(y))


def _make_score(mask_area: int, confidence: float, center_bias: float, frame_area: int) -> float:
    normalized_mask_area = mask_area / max(frame_area, 1)
    return (
        AREA_WEIGHT * normalized_mask_area
        + CONF_WEIGHT * confidence
        + CENTER_WEIGHT * center_bias
    )


# ------------------------------------------------------------------ face clustering

def _face_items_from_result(result, masks, boxes, count, frame_width, frame_height) -> list[dict]:
    items = []
    for index in range(count):
        class_id = int(_scalar(_get_row(boxes.cls, index), -1))
        if get_class_name(result, class_id) not in FACE_PART_CLASS_NAMES:
            continue
        mask = resize_mask_to_frame(masks.data[index], frame_width, frame_height)
        if compute_mask_area(mask) == 0:
            continue
        items.append({
            "index": index,
            "mask": mask,
            "bbox": _bbox_from_row(_get_row(boxes.xyxy, index, [0, 0, 0, 0])),
            "confidence": float(_scalar(_get_row(boxes.conf, index), 0.0)),
        })
    return items


def _cluster_face_items(items: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for item in items:
        matching = [
            i for i, cluster in enumerate(clusters)
            if any(
                np.logical_and(item["mask"], m["mask"]).any()
                or _bboxes_overlap_or_close(item["bbox"], m["bbox"])
                for m in cluster
            )
        ]
        if not matching:
            clusters.append([item])
            continue
        first = matching[0]
        clusters[first].append(item)
        for i in reversed(matching[1:]):
            clusters[first].extend(clusters.pop(i))
    return clusters


def _cluster_to_candidate(cluster: list[dict], frame_width: int, frame_height: int, frame_area: int) -> SelectedPerson | None:
    union_mask = np.zeros((frame_height, frame_width), dtype=bool)
    for item in cluster:
        union_mask = np.logical_or(union_mask, item["mask"])

    mask_area = compute_mask_area(union_mask)
    if mask_area == 0:
        return None

    bbox = _bbox_union([item["bbox"] for item in cluster])
    confidence = float(np.mean([item["confidence"] for item in cluster]))
    center_bias = compute_center_bias(compute_bbox_center(bbox), frame_width, frame_height)
    related_indices = tuple(item["index"] for item in cluster)

    return SelectedPerson(
        index=related_indices[0],
        bbox_xyxy=bbox,
        confidence=confidence,
        class_id=-1,
        class_name="face",
        mask=union_mask,
        mask_area=mask_area,
        center_bias=center_bias,
        score=_make_score(mask_area, confidence, center_bias, frame_area),
        candidate_type="face",
        related_indices=related_indices,
    )


def _build_face_candidates(result, frame_width: int, frame_height: int, count: int, frame_area: int) -> list[SelectedPerson]:
    items = _face_items_from_result(result, result.masks, result.boxes, count, frame_width, frame_height)
    clusters = _cluster_face_items(items)
    return [c for cluster in clusters if (c := _cluster_to_candidate(cluster, frame_width, frame_height, frame_area))]


# ------------------------------------------------------------------ person selection

def _is_person_candidate(class_id: int, class_name: str) -> bool:
    return class_id == 10 or class_name in PERSON_CLASS_NAMES


def _result_count(result) -> int:
    return min(
        _mask_count(result.masks),
        _box_count(result.boxes),
        _value_count(getattr(result.boxes, "conf", [])),
        _value_count(getattr(result.boxes, "cls", [])),
    )


def _build_person_candidate(result, index: int, frame_width: int, frame_height: int, frame_area: int, conf_threshold: float) -> SelectedPerson | None:
    boxes = result.boxes
    confidence = float(_scalar(_get_row(boxes.conf, index), 0.0))
    if confidence < conf_threshold:
        return None

    class_id = int(_scalar(_get_row(boxes.cls, index), -1))
    class_name = get_class_name(result, class_id)
    if not _is_person_candidate(class_id, class_name):
        return None

    bbox = _bbox_from_row(_get_row(boxes.xyxy, index, [0, 0, 0, 0]))
    mask = resize_mask_to_frame(result.masks.data[index], frame_width, frame_height)
    mask_area = compute_mask_area(mask) or _bbox_area(bbox)
    center_bias = compute_center_bias(compute_bbox_center(bbox), frame_width, frame_height)

    return SelectedPerson(
        index=index,
        bbox_xyxy=bbox,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        mask=mask,
        mask_area=mask_area,
        center_bias=center_bias,
        score=_make_score(mask_area, confidence, center_bias, frame_area),
        track_id=_track_id_at(boxes, index),
        candidate_type="person",
        related_indices=(index,),
    )


def select_main_person(result, frame_width: int, frame_height: int, conf_threshold: float = PERSON_CONF_THRESHOLD) -> SelectedPerson | None:
    if result is None or getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return None

    count = _result_count(result)
    if count <= 0:
        return None

    frame_area = max(frame_width * frame_height, 1)
    candidates = [
        c for i in range(count)
        if (c := _build_person_candidate(result, i, frame_width, frame_height, frame_area, conf_threshold))
    ]
    candidates.extend(_build_face_candidates(result, frame_width, frame_height, count, frame_area))

    return max(candidates, key=lambda p: p.score) if candidates else None


# ------------------------------------------------------------------ related masks

def _face_masks_for_selected(result, selected: SelectedPerson, count: int, frame_width: int, frame_height: int) -> list[np.ndarray]:
    masks = result.masks
    related = [
        resize_mask_to_frame(masks.data[i], frame_width, frame_height)
        for i in selected.related_indices
        if 0 <= i < count
    ]
    return related or [selected.mask]


def _is_related_face_mask(face_mask: np.ndarray, face_area: int, selected: SelectedPerson) -> bool:
    intersection = int(np.logical_and(face_mask, selected.mask).sum())
    overlap_ratio = intersection / max(face_area, 1)
    center = _mask_center(face_mask)
    center_inside = center is not None and _point_in_bbox(center, selected.bbox_xyxy)
    return overlap_ratio >= RELATED_MASK_MIN_OVERLAP or center_inside


def _person_related_masks(result, selected: SelectedPerson, count: int, frame_width: int, frame_height: int) -> list[np.ndarray]:
    masks, boxes = result.masks, result.boxes
    related = [selected.mask]
    for index in range(count):
        if index == selected.index:
            continue
        class_id = int(_scalar(_get_row(boxes.cls, index), -1))
        if get_class_name(result, class_id) not in FACE_PART_CLASS_NAMES:
            continue
        face_mask = resize_mask_to_frame(masks.data[index], frame_width, frame_height)
        face_area = compute_mask_area(face_mask)
        if face_area > 0 and _is_related_face_mask(face_mask, face_area, selected):
            related.append(face_mask)
    return related


def collect_related_masks(result, selected_person: SelectedPerson | None, frame_width: int, frame_height: int) -> list[np.ndarray]:
    if selected_person is None:
        return []
    if result is None or getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return []

    count = min(_mask_count(result.masks), _box_count(result.boxes), _value_count(getattr(result.boxes, "cls", [])))

    if selected_person.candidate_type == "face":
        return _face_masks_for_selected(result, selected_person, count, frame_width, frame_height)
    return _person_related_masks(result, selected_person, count, frame_width, frame_height)