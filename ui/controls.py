from dataclasses import dataclass

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

from core.display_mode import DEFAULT_DISPLAY_MODE, DisplayMode, display_mode_label
from core.settings import SAMPLES_DIR
from ui.components import SingleSelectDropdown, create_button
from ui.file_list import create_file_combo
from ui.styles import (
    BOTTOM_BAR_MARGINS,
    BOTTOM_BAR_MAX_HEIGHT,
    BOTTOM_BAR_SPACING,
)


@dataclass
class ControlPanel:
    widget: QWidget
    combo: QComboBox
    mode_dropdown: SingleSelectDropdown
    btn_start: QPushButton
    btn_stop: QPushButton


def create_control_panel() -> ControlPanel:
    bottom = QWidget()
    bottom.setMaximumHeight(BOTTOM_BAR_MAX_HEIGHT)

    bar = QHBoxLayout()
    bar.setContentsMargins(BOTTOM_BAR_MARGINS)
    bar.setSpacing(BOTTOM_BAR_SPACING)
    bottom.setLayout(bar)

    combo = create_file_combo(SAMPLES_DIR)
    bar.addWidget(combo, stretch=5)

    mode_dropdown = SingleSelectDropdown(
        [(display_mode_label(mode), mode) for mode in DisplayMode],
        default_value=DEFAULT_DISPLAY_MODE,
    )
    bar.addWidget(mode_dropdown, stretch=3)

    buttons = QWidget()
    buttons_layout = QHBoxLayout()
    buttons_layout.setContentsMargins(0, 0, 0, 0)
    buttons_layout.setSpacing(BOTTOM_BAR_SPACING)
    buttons.setLayout(buttons_layout)

    btn_start = create_button("Запустити")
    buttons_layout.addWidget(btn_start, stretch=1)

    btn_stop = create_button("Стоп")
    btn_stop.setEnabled(False)
    btn_stop.setVisible(False)
    buttons_layout.addWidget(btn_stop, stretch=1)

    bar.addWidget(buttons, stretch=2)

    return ControlPanel(bottom, combo, mode_dropdown, btn_start, btn_stop)


def set_tracking_controls_enabled(panel: ControlPanel, enabled: bool) -> None:
    panel.btn_start.setEnabled(enabled)
    panel.combo.setEnabled(enabled)


def set_stop_button_state(panel: ControlPanel, enabled: bool, visible: bool | None = None) -> None:
    panel.btn_stop.setEnabled(enabled)
    if visible is not None:
        panel.btn_stop.setVisible(visible)
