from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
)

from ui.styles import (
    CONTROL_BUTTON_STYLE,
    CONTROL_HEIGHT,
    CONTROL_MENU_ITEM_HEIGHT,
    CONTROL_COMBO_STYLE,
    DROP_AREA_HINT_STYLE,
    DROP_AREA_STYLE,
    DROP_AREA_TITLE_STYLE,
    FPS_LABEL_STYLE,
    MODE_MARK_DOT_SIZE,
    MODE_MARK_BORDER_COLOR,
    MODE_MARK_ICON_BOX_SIZE,
    MODE_MARK_SELECTED_COLOR,
    MODE_MARK_SIZE,
    VIDEO_LABEL_STYLE,
)


class FixedHeightItemDelegate(QStyledItemDelegate):
    def __init__(self, item_height: int, parent=None) -> None:
        super().__init__(parent)
        self.item_height = item_height

    def sizeHint(self, option, index) -> QSize:
        size = super().sizeHint(option, index)
        size.setHeight(self.item_height)
        return size


class SingleSelectDropdown(QComboBox):
    def __init__(self, options, default_value=None, parent=None) -> None:
        super().__init__(parent)
        self._empty_icon = _create_radio_icon(False)
        self._selected_icon = _create_radio_icon(True)
        self.setMinimumHeight(CONTROL_HEIGHT)
        self.setFixedHeight(CONTROL_HEIGHT)
        self.setIconSize(QSize(MODE_MARK_ICON_BOX_SIZE, MODE_MARK_ICON_BOX_SIZE))
        self.setItemDelegate(FixedHeightItemDelegate(CONTROL_MENU_ITEM_HEIGHT, self))
        self.setStyleSheet(CONTROL_COMBO_STYLE)

        for label, value in options:
            self.addItem(self._empty_icon, label, value)

        if default_value is not None:
            index = self.findData(default_value)
            if index >= 0:
                self.setCurrentIndex(index)

        self.currentIndexChanged.connect(self._update_selection_icons)
        self._update_selection_icons()

    def currentValue(self):
        return self.currentData()

    def _update_selection_icons(self) -> None:
        current_index = self.currentIndex()
        for index in range(self.count()):
            icon = self._selected_icon if index == current_index else self._empty_icon
            self.setItemIcon(index, icon)


class FileDropArea(QFrame):
    files_dropped = pyqtSignal(list)
    select_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet(DROP_AREA_STYLE)
        self.setMinimumHeight(260)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)
        layout.setContentsMargins(32, 32, 32, 32)
        self.setLayout(layout)

        title = QLabel("Перетягніть сюди зображення або відео")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(DROP_AREA_TITLE_STYLE)
        layout.addWidget(title)

        select_button = QPushButton("або оберіть файл вручну")
        select_button.setStyleSheet(DROP_AREA_HINT_STYLE)
        select_button.clicked.connect(self.select_requested.emit)
        layout.addWidget(select_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def dragEnterEvent(self, event) -> None:
        if self._has_local_files(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._has_local_files(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    @staticmethod
    def _has_local_files(mime_data) -> bool:
        return mime_data.hasUrls() and any(url.isLocalFile() for url in mime_data.urls())


def create_video_label() -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(VIDEO_LABEL_STYLE)
    return label


def create_fps_label(parent: QLabel) -> QLabel:
    label = QLabel("FPS: —", parent=parent)
    label.setVisible(False)
    label.setStyleSheet(FPS_LABEL_STYLE)
    label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    return label


def create_checkbox(text: str, checked: bool = False) -> QCheckBox:
    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)
    return checkbox


def create_button(text: str, fixed_width: int | None = None) -> QPushButton:
    button = QPushButton(text)
    button.setMinimumHeight(CONTROL_HEIGHT)
    button.setFixedHeight(CONTROL_HEIGHT)
    button.setStyleSheet(CONTROL_BUTTON_STYLE)
    if fixed_width is not None:
        button.setFixedWidth(fixed_width)
    return button


def _create_radio_icon(selected: bool) -> QIcon:
    pixmap = QPixmap(MODE_MARK_ICON_BOX_SIZE, MODE_MARK_ICON_BOX_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(MODE_MARK_BORDER_COLOR), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    circle_offset = (MODE_MARK_ICON_BOX_SIZE - MODE_MARK_SIZE) / 2
    painter.drawEllipse(QRectF(circle_offset, circle_offset, MODE_MARK_SIZE, MODE_MARK_SIZE))

    if selected:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(MODE_MARK_SELECTED_COLOR))
        dot_offset = (MODE_MARK_ICON_BOX_SIZE - MODE_MARK_DOT_SIZE) / 2
        painter.drawEllipse(QRectF(dot_offset, dot_offset, MODE_MARK_DOT_SIZE, MODE_MARK_DOT_SIZE))

    painter.end()
    return QIcon(pixmap)
