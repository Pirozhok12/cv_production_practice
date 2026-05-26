from enum import Enum


class DisplayMode(Enum):
    ORIGINAL = "original"
    DEFAULT = "default"
    CORTICAL_VISION = "cortical_vision"
    ALL = "all"


DISPLAY_MODE_LABELS = {
    DisplayMode.ORIGINAL: "Вхідне зображення",
    DisplayMode.DEFAULT: "Без кортикального зору",
    DisplayMode.CORTICAL_VISION: "Кортикальний зір",
    DisplayMode.ALL: "Усе",
}

DEFAULT_DISPLAY_MODE = DisplayMode.CORTICAL_VISION


def display_mode_label(mode: DisplayMode) -> str:
    return DISPLAY_MODE_LABELS[mode]
