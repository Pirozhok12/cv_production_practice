from PyQt6.QtCore import QMargins


WINDOW_TITLE = "cv"
WINDOW_SIZE = (1200, 800)
WINDOW_MIN_SIZE = (400, 300)

ROOT_MARGINS = (0, 0, 0, 0)
ROOT_SPACING = 0

PREVIEW_SIZE = 48
CONTROL_HEIGHT = 48
CONTROL_MENU_ITEM_HEIGHT = 44
FILE_MENU_ITEM_HEIGHT = 56
CONTROL_HORIZONTAL_PADDING = 12
MODE_MARK_SIZE = 14
MODE_MARK_ICON_BOX_SIZE = 18
MODE_MARK_DOT_SIZE = 6
BOTTOM_BAR_MAX_HEIGHT = 64
BOTTOM_BAR_MARGINS = QMargins(12, 8, 12, 8)
BOTTOM_BAR_SPACING = 10

FPS_MARGIN = 10
MODE_MARK_BORDER_COLOR = "#777777"
MODE_MARK_SELECTED_COLOR = "#d71920"

VIDEO_LABEL_STYLE = "background: black;"
FPS_LABEL_STYLE = (
    "color: #00ff88; background: rgba(0,0,0,160);"
    "padding: 2px 8px; border-radius: 4px; font-weight: bold;"
)

CONTROL_COMBO_STYLE = f"""
QComboBox {{
    padding: 0 {CONTROL_HORIZONTAL_PADDING + 16}px 0 {CONTROL_HORIZONTAL_PADDING}px;
}}
QComboBox QAbstractItemView {{
    padding: 4px 0;
    outline: 0;
}}
"""

CONTROL_BUTTON_STYLE = f"""
QPushButton {{
    padding: 0 {CONTROL_HORIZONTAL_PADDING}px;
}}
"""

DROP_AREA_STYLE = """
QFrame {
    background: #111111;
    border: 2px dashed #777777;
    border-radius: 8px;
}
"""

DROP_AREA_TITLE_STYLE = """
color: #f2f2f2;
font-size: 24px;
font-weight: 600;
background: transparent;
border: none;
"""

DROP_AREA_HINT_STYLE = """
color: #c8c8c8;
font-size: 15px;
background: transparent;
border: none;
"""
