from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
import numpy as np

class Push_Btn(QtWidgets.QPushButton):
    def __init__(self, idx, signal) -> None:
        super().__init__()
        self.idx = idx
        self.setText("Push")
        self.clicked.connect(lambda: signal.emit(self.idx))

class Param_window(QtWidgets.QMainWindow):
    put_to_phone_signal = pyqtSignal(int)
    
    def setup(self, popsize, param_change_num, ans=None, IQM_names=[]):
        self.popsize = popsize
        self.param_change_num = param_change_num
        self.ans = ans
        self.move(100, 100)

        self.fitness = [-1]*popsize
        self.label_trial_denorm = []
        self.label_score = []
        self.label_IQM = []

        # title
        label = QtWidgets.QLabel(self.centralwidget)
        label.setText("param value")
        label.setStyleSheet("background-color: rgb(0, 51, 102);")
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(label, 0, 0, 1, param_change_num)

        label = QtWidgets.QLabel(self.centralwidget)
        label.setText("score")
        label.setStyleSheet("background-color: rgb(0, 51, 102);")
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(label, 0, param_change_num, 1, 1)

        # IQM_names = []
        # IQM_names = ["sharpness", "noise"]

        for i, IQM_name in enumerate(IQM_names):
            label = QtWidgets.QLabel(self.centralwidget)
            label.setText(IQM_name)
            label.setStyleSheet("background-color: rgb(0, 51, 102);")
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(label, 0, param_change_num+1+i, 1, 1)

        if len(IQM_names):
            label = QtWidgets.QLabel(self.centralwidget)
            label.setText("推到手機")
            label.setStyleSheet("background-color: rgb(0, 51, 102);")
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(label, 0, param_change_num+1+len(IQM_names), 1, 1)

        # score label
        for i in range(popsize):
            label_trial_denorm = []
            label_IQM = []
            for j in range(param_change_num):
                label = QtWidgets.QLabel(self.centralwidget)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label_trial_denorm.append(label)
                self.gridLayout.addWidget(label, i+1, j, 1, 1)
            self.label_trial_denorm.append(label_trial_denorm)

            label = QtWidgets.QLabel(self.centralwidget)
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(label, i+1, param_change_num, 1, 1)
            self.label_score.append(label)

            for j in range(len(IQM_names)):
                label = QtWidgets.QLabel(self.centralwidget)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label_IQM.append(label)
                self.gridLayout.addWidget(label, i+1, self.param_change_num+1+j, 1, 1)
            self.label_IQM.append(label_IQM)

            if len(IQM_names):
                self.gridLayout.addWidget(Push_Btn(i, self.put_to_phone_signal), i+1, param_change_num+1+len(IQM_names), 1, 1)


    def __init__(self, ):
        super().__init__()
    
        self.setWindowTitle("param window")
        self.resize(0, 0)
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayout_parent = QtWidgets.QVBoxLayout(self.centralwidget)
        self.gridLayout = QtWidgets.QGridLayout()
     
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setColumnStretch(3, 1)
        self.gridLayout.setColumnStretch(4, 1)
        self.verticalLayout_parent.addLayout(self.gridLayout)
        self.setCentralWidget(self.centralwidget)

        self.setStyleSheet("QMainWindow {background-color: rgb(54, 69, 79);}"
                           """
                                QLabel {
                                    font-size:10pt; font-family:微軟正黑體; font-weight: bold;
                                    color: white;
                                    border: 1px solid black;
                                    padding: 3px;
                                }
                                QToolTip { 
                                    background-color: black; 
                                    border: black solid 1px
                                }
                                QPushButton{
                                    font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);
                                }
                                """
                           )

    def update(self, idx, trial, trial_denorm, score, IQM=[]):
        self.fitness[idx] = score
        self.label_score[idx].setText(str(np.round(score, 5)))
        if isinstance(self.ans, np.ndarray): color = 255*(1-np.abs(trial-self.ans))
        for j in range(self.param_change_num):
            self.label_trial_denorm[idx][j].setText(str(np.round(trial_denorm[j], 4)))
            if isinstance(self.ans, np.ndarray):
                self.label_trial_denorm[idx][j].setStyleSheet("color: rgb(255, {}, 255)".format(color[j]))

        order = np.argsort(self.fitness)
        color = 255 - np.arange(0, 150, 150/self.popsize)
        for i, c in zip(order, color):
            self.label_score[i].setStyleSheet("color: rgb({0}, {0}, {0})".format(c))

        for j in range(len(IQM)):
            self.label_IQM[idx][j].setText(str(np.round(IQM[j], 4)))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    IQM_names = []
    # IQM_names = ["sharpness", "noise"]
    popsize = 15
    param_change_num = 16
    w = Param_window()
    w.setup(popsize=popsize, param_change_num=param_change_num, ans=[0]*param_change_num, IQM_names=IQM_names)
    pop = np.random.rand(popsize, param_change_num)
    for i in range(popsize):
        w.update(i, pop[i], np.random.rand(), np.random.rand(len(IQM_names)))
    w.show()

    # w = Param_window() # 重新創新視窗才不會讓字重疊
    # w.setup(popsize=popsize, param_change_num=param_change_num, ans=[0]*param_change_num, IQM_names=IQM_names)
    # pop = np.random.rand(popsize, param_change_num)
    # for i in range(popsize):
    #     w.update(i, pop[i], np.random.rand(), np.random.rand(len(IQM_names)))
    # w.show()
    sys.exit(app.exec_())