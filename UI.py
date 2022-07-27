# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.Qt import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QSize
import numpy as np


class ImageViewer(QtWidgets.QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation

    def __init__(self, pixmap=None):
        super().__init__()
        self.setPixmap(pixmap)

    def setPixmap(self, pixmap):
        if self.pixmap != pixmap:
            self.pixmap = pixmap
            if isinstance(pixmap, QPixmap):
                self._sizeHint = pixmap.size()
            else:
                self._sizeHint = QSize()
            self.updateGeometry()
            self.updateScaled()

    def setAspectRatio(self, ratio):
        if self.ratio != ratio:
            self.ratio = ratio
            self.updateScaled()

    def setTransformation(self, transformation):
        if self.transformation != transformation:
            self.transformation = transformation
            self.updateScaled()

    def updateScaled(self):
        if self.pixmap:
            self.scaled = self.pixmap.scaled(
                self.size(), self.ratio, self.transformation)
        self.update()

    def sizeHint(self):
        return self._sizeHint

    def resizeEvent(self, event):
        self.updateScaled()

    def paintEvent(self, event):
        if not self.pixmap:
            return
        qp = QPainter(self)
        r = self.scaled.rect()
        r.moveCenter(self.rect().center())
        qp.drawPixmap(r, self.scaled)


class ParamModifyBlock(QtWidgets.QVBoxLayout):

    def __init__(self, parent, title, name=["Y", "Chroma"], col=[3, 4]):
        super().__init__()
        self.title = title
        self. name = name
        self.checkBoxes_title = []
        self.checkBoxes = []
        self.lineEdits = []

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.setHorizontalSpacing(7)

        title_wraper = QtWidgets.QHBoxLayout()
        label_title = QtWidgets.QLabel(parent)
        label_title.setText(title)
        label_title.setStyleSheet("background-color:rgb(72, 72, 72);")
        title_wraper.addWidget(label_title)

        self.checkBoxes_title = QtWidgets.QCheckBox(parent)
        title_wraper.addWidget(self.checkBoxes_title)

        # title_wraper.setStretch(0, 1)
        # title_wraper.setStretch(1, 0)
        # container.setStyleSheet("background-color:rgb(72, 72, 72);")

        idx = len(self.checkBoxes)
        for i in range(sum(col)):
            checkBox = QtWidgets.QCheckBox(parent)
            checkBox.setToolTip("打勾代表將值固定")
            self.checkBoxes.append(checkBox)

            lineEdit = QtWidgets.QLineEdit(parent)
            self.lineEdits.append(lineEdit)

        for i in range(len(col)):
            label_name = QtWidgets.QLabel(parent)
            label_name.setText(name[i])
            label_name.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
            gridLayout.addWidget(label_name, i, 0)

            for j in range(col[i]):
                gridLayout.addWidget(self.checkBoxes[idx], i, 2+j*2)
                gridLayout.addWidget(self.lineEdits[idx], i, 1+j*2)
                idx += 1

        gridLayout.setColumnStretch(0, 1)

        self.addLayout(title_wraper)
        self.addLayout(gridLayout)

        self.checkBoxes_title.clicked.connect(self.toggle_checkBoxes_title)

    def toggle_checkBoxes_title(self):
        checked = self.checkBoxes_title.isChecked()
        for box in self.checkBoxes:
            box.setChecked(checked)


class Ui_MainWindow(object):

    def param_range_block(self, parent, title, name=["Y", "Chroma"], row=2):
        verticalLayout = QtWidgets.QVBoxLayout()

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.setHorizontalSpacing(7)

        label_title = QtWidgets.QLabel(parent)
        label_title.setText(title)
        label_title.setStyleSheet("background-color:rgb(72, 72, 72);")

        l1 = QtWidgets.QLabel(parent)
        l2 = QtWidgets.QLabel(parent)
        l1.setText("預設範圍")
        l2.setText("自訂範圍")
        gridLayout.addWidget(l1, 0, 1)
        gridLayout.addWidget(l2, 0, 2)

        idx = len(self.label_defult_range)
        for i in range(row):
            label = QtWidgets.QLabel(parent)
            label.setText("#")
            self.label_defult_range.append(label)

            lineEdit = QtWidgets.QLineEdit(parent)
            self.lineEdits_range.append(lineEdit)

        for i in range(1, row+1):
            label_name = QtWidgets.QLabel(parent)
            label_name.setText(name[i-1])
            label_name.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
            gridLayout.addWidget(label_name, i, 0)

            gridLayout.addWidget(self.label_defult_range[idx], i, 1)
            gridLayout.addWidget(self.lineEdits_range[idx], i, 2)
            idx += 1

        gridLayout.setColumnStretch(0, 2)
        gridLayout.setColumnStretch(1, 3)
        gridLayout.setColumnStretch(2, 3)

        verticalLayout.addWidget(label_title)
        verticalLayout.addLayout(gridLayout)

        return verticalLayout

    def param_setting_block(self, parent):
        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.setHorizontalSpacing(7)

        text = ["population size", "generations"]
        self.hyper_param_title = text
        for i in range(len(text)):
            label = QtWidgets.QLabel(parent)
            label.setText(text[i])

            lineEdit = QtWidgets.QLineEdit(parent)
            # label.setToolTip(tip[i])
            self.lineEdits_hyper_setting.append(lineEdit)

            gridLayout.addWidget(label, i, 0)
            gridLayout.addWidget(lineEdit, i, 1)

        return gridLayout

    def tab1_block(self):
        tab = QtWidgets.QWidget()

        parentGridLayout = QtWidgets.QGridLayout(tab)
        gridLayout = QtWidgets.QGridLayout()
        horizontalLayout = QtWidgets.QHBoxLayout()

        # upper
        self.btn_select_project = QtWidgets.QPushButton(tab)
        self.btn_select_project.setText("選擇project")
        self.btn_select_project.setToolTip("選擇tuning project folder")
        gridLayout.addWidget(self.btn_select_project, 0, 0)

        self.btn_select_exe = QtWidgets.QPushButton(tab)
        self.btn_select_exe.setText("選擇ParameterParser")
        gridLayout.addWidget(self.btn_select_exe, 1, 0)

        label = QtWidgets.QLabel(tab)
        label.setText("bin檔名稱")
        gridLayout.addWidget(label, 2, 0)

        self.label_project_path = QtWidgets.QLabel(tab)
        self.label_project_path.setText("")
        gridLayout.addWidget(self.label_project_path, 0, 1)

        self.label_exe_path = QtWidgets.QLabel(tab)
        self.label_exe_path.setText("")
        gridLayout.addWidget(self.label_exe_path, 1, 1)

        self.lineEdits_bin_name = QtWidgets.QLineEdit(tab)
        gridLayout.addWidget(self.lineEdits_bin_name, 2, 1)

        # medium
        self.btn_select_ROI = QtWidgets.QPushButton(tab)
        self.btn_select_ROI.setText("選擇ROI")
        self.btn_select_ROI.setToolTip("會拍攝一張照片並選取範圍")
        horizontalLayout.addWidget(self.btn_select_ROI)

        label = QtWidgets.QLabel(tab)
        label.setText("請選取要計算的區域，選好後按space或enter鍵確定，按c取消")
        horizontalLayout.addWidget(label)

        # lower
        self.label_ROI_img = ImageViewer()
        self.label_ROI_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ROI_img.setText("ROI圖片")
        self.label_ROI_img.setStyleSheet("background-color: rgb(0, 0, 0);")

        parentGridLayout.addLayout(gridLayout, 0, 0, 1, 1)
        parentGridLayout.addLayout(horizontalLayout, 1, 0, 1, 1)
        parentGridLayout.addWidget(self.label_ROI_img, 2, 0, 1, 1)

        horizontalLayout.setStretch(0, 1)
        horizontalLayout.setStretch(1, 8)
        gridLayout.setColumnStretch(0, 1)
        gridLayout.setColumnStretch(1, 8)
        parentGridLayout.setRowStretch(2, 1)

        tab.setStyleSheet("QLabel{font-size:12pt; font-family:微軟正黑體; color:white;}"
                          "QPushButton{font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}"
                          "QLineEdit{font-size:12pt; font-family:微軟正黑體; background-color: rgb(255, 255, 255); border: 2px solid gray; border-radius: 5px;}")

        return tab

    def tab2_block(self):
        tab = QtWidgets.QWidget()

        horizontalLayout = QtWidgets.QHBoxLayout(tab)
        horizontalLayout.setObjectName("horizontalLayout")

        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.setContentsMargins(0, 0, 0, 0)
        # self.verticalLayout.addItem(spacerItem)
        self.ParamModifyBlock.append(ParamModifyBlock(
            tab, "Noise Profile", ["Y", "Cb", "Cr"], col=[3, 4, 4]))
        self.ParamModifyBlock.append(ParamModifyBlock(tab, "Denoise Scale"))
        self.ParamModifyBlock.append(
            ParamModifyBlock(tab, "Denoise Edge Softness"))
        self.ParamModifyBlock.append(ParamModifyBlock(tab, "Denoise Weight"))
        verticalLayout.addLayout(self.ParamModifyBlock[0])
        verticalLayout.addLayout(self.ParamModifyBlock[1])
        verticalLayout.addLayout(self.ParamModifyBlock[2])
        verticalLayout.addLayout(self.ParamModifyBlock[3])
        self.label_ans = QtWidgets.QLabel(tab)
        self.label_ans.setText('ans')
        verticalLayout.addWidget(self.label_ans)
        verticalLayout.addItem(spacerItem)
        horizontalLayout.addLayout(verticalLayout)

        verticalLayout_2 = QtWidgets.QVBoxLayout()
        # self.verticalLayout_2.addItem(spacerItem)
        verticalLayout_2.addLayout(self.param_range_block(
            tab, "Noise Profile", ["Y", "Cb", "Cr"], row=3))
        verticalLayout_2.addLayout(
            self.param_range_block(tab, "Denoise Scale"))
        verticalLayout_2.addLayout(
            self.param_range_block(tab, "Denoise Edge Softness"))
        verticalLayout_2.addLayout(
            self.param_range_block(tab, "Denoise Weight"))
        verticalLayout_2.addItem(spacerItem)
        horizontalLayout.addLayout(verticalLayout_2)

        verticalLayout_3 = QtWidgets.QVBoxLayout()
        # self.verticalLayout_3.addItem(spacerItem)
        verticalLayout_3.addLayout(self.param_setting_block(tab))
        verticalLayout_3.addItem(spacerItem)
        horizontalLayout.addLayout(verticalLayout_3)

        horizontalLayout.setStretch(0, 3)
        horizontalLayout.setStretch(1, 2)
        horizontalLayout.setStretch(2, 2)

        tab.setStyleSheet("QLabel{font-size:12pt; font-family:微軟正黑體; color:white;}"
                          """
                        QLineEdit{
                            background-color: rgb(255, 255, 255); 
                            border: 2px solid gray; 
                            border-radius: 5px;
                            font-size:12pt;
                            font-family:微軟正黑體;
                        }
                        """
                          )

        return tab

    def tab3_block(self):
        tab = QtWidgets.QWidget()

        # parent
        gridLayout = QtWidgets.QGridLayout(tab)

        # upper
        horizontalLayout = QtWidgets.QHBoxLayout()

        self.btn_run = QtWidgets.QPushButton(tab)
        self.btn_run.setText("Run")
        self.btn_run.setStyleSheet(
            "font-family:Agency FB; font-size:30pt; width: 100%; height: 100%;")
        horizontalLayout.addWidget(self.btn_run)

        gridLayout_ML = QtWidgets.QGridLayout()

        self.pretrain_model = QtWidgets.QCheckBox(tab)
        gridLayout_ML.addWidget(self.pretrain_model, 0, 0, 1, 1)

        self.train = QtWidgets.QCheckBox(tab)
        gridLayout_ML.addWidget(self.train, 1, 0, 1, 1)

        label = QtWidgets.QLabel(tab)
        label.setText("pretrain model")
        gridLayout_ML.addWidget(label, 0, 1, 1, 1)

        label = QtWidgets.QLabel(tab)
        label.setText("training")
        gridLayout_ML.addWidget(label, 1, 1, 1, 1)

        horizontalLayout.addLayout(gridLayout_ML)

        label = QtWidgets.QLabel(tab)
        label.setText("總分")
        horizontalLayout.addWidget(label)

        self.label_score = QtWidgets.QLabel(tab)
        self.label_score.setText("#")
        horizontalLayout.addWidget(self.label_score)

        gridLayout_gen = QtWidgets.QGridLayout()

        label = QtWidgets.QLabel(tab)
        label.setText("generation:")
        gridLayout_gen.addWidget(label, 0, 0, 1, 1)

        label = QtWidgets.QLabel(tab)
        label.setText("individual:")
        gridLayout_gen.addWidget(label, 1, 0, 1, 1)

        self.label_generation = QtWidgets.QLabel(tab)
        self.label_generation.setText("#")
        gridLayout_gen.addWidget(self.label_generation, 0, 1, 1, 1)

        self.label_individual = QtWidgets.QLabel(tab)
        self.label_individual.setText("#")
        gridLayout_gen.addWidget(self.label_individual, 1, 1, 1, 1)

        horizontalLayout.addLayout(gridLayout_gen)

        self.label_time = QtWidgets.QLabel(tab)
        self.label_time.setText("Time")
        self.label_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_time.setStyleSheet(
            "font-family:Agency FB; font-size:30pt; width: 100%; height: 100%;")
        horizontalLayout.addWidget(self.label_time)

        # horizontalLayout.setStretch(1, 2)
        # horizontalLayout.setStretch(3, 1)
        # horizontalLayout.setStretch(4, 1)
        # horizontalLayout.setStretch(5, 1)
        horizontalLayout.setContentsMargins(7, 0, 7, -1)
        horizontalLayout.setSpacing(10)

        gridLayout.addLayout(horizontalLayout, 0, 0, 1, 1)

        tabWidget_plot = QtWidgets.QTabWidget(tab)
        tab_score = QtWidgets.QWidget()
        plot_wraprt = QtWidgets.QVBoxLayout(tab_score)
        self.label_best_score_plot = QtWidgets.QLabel(tab_score)
        self.label_best_score_plot.setText("分數圖")
        self.label_best_score_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_best_score_plot.setStyleSheet(
            "background-color:rgb(0, 0, 0)")
        plot_wraprt.addWidget(self.label_best_score_plot)
        tabWidget_plot.addTab(tab_score, "分數圖")

        tab_hyper = QtWidgets.QWidget()
        plot_wraprt = QtWidgets.QVBoxLayout(tab_hyper)
        self.label_hyper_param_plot = QtWidgets.QLabel(tab_hyper)
        self.label_hyper_param_plot.setText("超參數")
        self.label_hyper_param_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_hyper_param_plot.setStyleSheet(
            "background-color:rgb(0, 0, 0)")
        plot_wraprt.addWidget(self.label_hyper_param_plot)
        tabWidget_plot.addTab(tab_hyper, "超參數")

        tab_loss = QtWidgets.QWidget()
        plot_wraprt = QtWidgets.QVBoxLayout(tab_loss)
        self.label_loss_plot = QtWidgets.QLabel(tab_loss)
        self.label_loss_plot.setText("loss")
        self.label_loss_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_loss_plot.setStyleSheet("background-color:rgb(0, 0, 0)")
        plot_wraprt.addWidget(self.label_loss_plot)
        tabWidget_plot.addTab(tab_loss, "loss")

        tab_update = QtWidgets.QWidget()
        plot_wraprt = QtWidgets.QVBoxLayout(tab_update)
        self.label_update_plot = QtWidgets.QLabel(tab_update)
        self.label_update_plot.setText("update rate")
        self.label_update_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_update_plot.setStyleSheet("background-color:rgb(0, 0, 0)")
        plot_wraprt.addWidget(self.label_update_plot)
        tabWidget_plot.addTab(tab_update, "update rate")

        gridLayout.addWidget(tabWidget_plot, 1, 0, 1, 1)
        # gridLayout.addWidget(self.label_best_score_plot, 1, 0, 1, 1)

        gridLayout.setRowStretch(1, 1)

        tab.setStyleSheet("QLabel{font-size:12pt; font-family:微軟正黑體; color:white;}"
                          "QPushButton{font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")

        return tab

    def setupUi(self, MainWindow):
        self.ParamModifyBlock = []

        self.label_defult_range = []
        self.lineEdits_range = []

        self.lineEdits_hyper_setting = []

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.Layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.Layout.setObjectName("horizontalLayout")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        # self.tabWidget.addTab(self.tab1_block(), "選擇project")
        self.tabWidget.addTab(self.tab2_block(), "參數設定")
        self.tabWidget.addTab(self.tab3_block(), "執行")
        self.tabWidget.setStyleSheet("font-size:12pt; font-family:微軟正黑體;")

        self.Layout.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.setStyleSheet("color: white")
        self.statusbar.showMessage('只存在5秒的消息', 5000)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        MainWindow.setStyleSheet(
            "*{background-color: rgb(124, 124, 124);}"
            """
            QMessageBox QLabel {
                font-size:12pt; font-family:微軟正黑體; color:white;
            }

            QMessageBox QPushButton{
                font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);
            }
            """

        )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Ackley"))


if __name__ == "__main__":
    import sys
    QtCore.QCoreApplication.setAttribute(
        Qt.AA_EnableHighDpiScaling)  # 適應windows縮放
    QtGui.QGuiApplication.setAttribute(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)  # 設置支持小數放大比例（適應如125%的縮放比）

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
