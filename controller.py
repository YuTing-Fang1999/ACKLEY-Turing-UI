from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from UI import Ui_MainWindow
from Param_window import Param_window

from Setting import Setting
from Tuning import Tuning
from Capture import Capture

import threading
import torch


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setting = Setting(self.ui)
        self.capture = Capture(self.ui)
        self.tuning = Tuning(self.ui, self.setting, self.capture)
        self.param_window = None

        self.setup_control()

    def setup_control(self):
        # self.ui.btn_select_project.clicked.connect(self.select_project)
        # self.ui.btn_select_exe.clicked.connect(self.select_exe)
        # self.ui.btn_select_ROI.clicked.connect(self.select_ROI)
        self.ui.btn_run.clicked.connect(self.run)
        self.capture.capture_fail_signal.connect(self.capture_fail)
        self.tuning.show_param_window_signal.connect(self.show_param_window)
        self.tuning.update_param_window_signal.connect(self.update_param_window)
        self.tuning.reset_param_window_signal.connect(self.reset_param_window)

        self.ui.closeEvent = lambda event: self.closeEvent(event)

    def closeEvent(self, event):
        print('window close')
        # ret = QMessageBox.information(self,"","確定要關閉嗎", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        # if ret == QMessageBox.Yes:
        self.setting.write_setting()
        if self.tuning.model: torch.save(self.tuning.model.state_dict(), "My_Model")
        if self.param_window: self.param_window.close()

        # if ret == QMessageBox.No:  # continue run
        # event.ignore()

    def select_ROI(self):
        roi = self.capture.select_ROI()
        self.setting.params['roi'] = roi
        print(roi)

    def capture_fail(self):
        QMessageBox.about(self, "拍攝未成功", "拍攝未成功\n請多按幾次拍照鍵測試\n再按ok鍵重新拍攝")
        self.capture.state.acquire()
        self.capture.state.notify()  # Unblock self if waiting.
        self.capture.state.release()

    def reset_param_window(self, popsize, param_change_num, ans):
        self.param_window = Param_window(popsize, param_change_num, ans)

    def show_param_window(self):
        self.param_window.show()

    def update_param_window(self, idx, param_value, score):
        self.param_window.update(idx, param_value, score)

    def run(self):
        if self.tuning.is_run:
            ret = QMessageBox.information(
                self, "", "確定要停止嗎?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret == QMessageBox.Yes:  # stop
                self.tuning.is_run = False
                self.ui.btn_run.setText('Run')

            if ret == QMessageBox.No:  # continue run
                pass

        else:
            # 更新設定的參數
            if not self.setting.set_param():
                return
            # UI的轉換
            self.ui.btn_run.setText('Stop')
            self.tuning.is_run = True

            # self.tuning.run(self.tuning_task_down)

            # Test
            self.tuning_task = threading.Thread(
                target=lambda: self.tuning.run_Ackley(self.tuning_task_down))
            # 建立一個子執行緒
            # self.tuning_task = threading.Thread(target = lambda: self.tuning.run(self.tuning_task_down))
            # 當主程序退出，該執行緒也會跟著結束
            self.tuning_task.daemon = True
            # 執行該子執行緒
            self.tuning_task.start()

    def tuning_task_down(self):
        self.tuning.is_run = False
        self.ui.btn_run.setText('Run')

    def select_project(self):
        folder_path = QFileDialog.getExistingDirectory(self,
                                                       "選擇project",
                                                       "./")                 # start path
        if folder_path == "":
            return
        self.setting.set_project(folder_path)

    def select_exe(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "選擇ParameterParser",
                                                         "./")                 # start path
        if filename == "":
            return
        self.ui.label_exe_path.setText(filename)
        self.setting.params["exe_path"] = filename
