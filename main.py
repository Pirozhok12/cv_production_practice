import sys
from PyQt6.QtWidgets import QApplication
from ui.app import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



    # TODO FPS\ALL_ITEMS_VIEVING