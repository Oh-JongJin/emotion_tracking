import sys
from setup import Ui_Dialog
from PyQt5.QtWidgets import QApplication


app = QApplication(sys.argv)
window = Ui_Dialog()
window.show()
sys.exit(app.exec())
