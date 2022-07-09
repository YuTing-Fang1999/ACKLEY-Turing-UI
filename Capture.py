import os
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

import threading


class Capture(QWidget):
    capture_fail_signal = pyqtSignal()

    def __init__(self, ui):
        super().__init__()

        self.ui = ui
        self.CAMERA_DEBUG = False
        self.CAMERA_PATH = '/sdcard/DCIM/Camera/'
        self.capture_num = 10
        self.state = threading.Condition()

    def capture(self, path="", focus_time=4, save_time=1, num=0):
        self.capture_fail_signal.emit()
        self.state.acquire()
        self.state.wait()
        self.state.release()
