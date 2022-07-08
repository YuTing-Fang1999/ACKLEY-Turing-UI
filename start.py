from PyQt5 import QtCore, QtWidgets


from controller import MainWindow_controller

import sys
# 自適應分辨率
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app = QtWidgets.QApplication(sys.argv)
window = MainWindow_controller()
window.show()
sys.exit(app.exec_())