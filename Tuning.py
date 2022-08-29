# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
import threading
from time import sleep
import math
import numpy as np
import json
import os
import random

from PyQt5.QtWidgets import QHBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


import matplotlib
matplotlib.use('Qt5Agg')

from ML import ML 

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=80):
        self.fig = Figure(figsize=(100, 30), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MplCanvas_timing():

    def __init__(self, ui, color, label):
        self.data = []

        self.canvas = MplCanvas()
        self.layout = QHBoxLayout(ui)

        self.color = color
        self.label = label

    def reset(self):
        self.data = []
        # The new widget is deleted when its parent is deleted.
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

    def update(self, data):
        self.data.append(data)
        self.canvas.axes.cla()

        lines = np.array(self.data).T
        x = list(range(lines.shape[-1]))
        for i in range(lines.shape[0]):
            self.canvas.axes.plot(x, lines[i], color=self.color[i], label=self.label[i])
        self.canvas.axes.legend(fontsize=15)
        self.canvas.fig.canvas.draw()  # 這裡注意是畫布重繪，self.figs.canvas
        self.canvas.fig.canvas.flush_events()  # 畫布刷新self.figs.canvas
        self.layout.addWidget(self.canvas)

        if len(self.data)>=600: self.data = []


class HyperOptimizer():
    def __init__(self, init_value, final_value, method, rate=0, decay_value=0):
        self.init_value = init_value
        self.final_value = final_value

        self.decay_value = decay_value
        self.rate = rate

        if method == "step":
            self.method = self.step_decay
        if method == "exponantial":
            self.method = self.exponantial_decay
        if method == "exponantial_reverse":
            self.method = self.exponantial_reverse
        if method == "constant":
            self.method = self.constant

    def constant(self, generation):
        return self.init_value

    def step_decay(self, generation):
        v = self.init_value - self.decay_value * generation
        return v

    def exponantial_decay(self, generation):
        # if generation % 15 == 0:
        #     self.final_value+=0.1
        # self.final_value-=0.1

        v = (self.init_value-self.final_value) * np.exp(-self.rate * (generation % 15)) + self.final_value

        return min(v, self.init_value)

    def exponantial_reverse(self, generation):
        # if generation % 15 == 0:
        #     self.init_value += 0.1
        #     self.rate -= 0.01

        v = self.init_value + (self.rate*(generation % 15))**np.exp(0.5)

        return min(v, self.final_value)

    def update(self, generation):
        return self.method(generation)


class Tuning(QWidget):  # 要繼承QWidget才能用pyqtSignal!!
    show_param_window_signal = pyqtSignal()
    setup_param_window_signal = pyqtSignal(int, int, np.ndarray)
    update_param_window_signal = pyqtSignal(int, np.ndarray, float, np.ndarray)


    def __init__(self, ui, setting, capture):
        super().__init__()
        self.ML = None

        self.ui = ui
        self.setting = setting
        self.capture = capture
        self.is_run = False

        # plot
        self.bset_score_plot = MplCanvas_timing(
            self.ui.label_best_score_plot, color=['r', 'g'], label=['score'])
        self.hyper_param_plot = MplCanvas_timing(
            self.ui.label_hyper_param_plot, color=['g', 'r'], label=['F', 'Cr'])
        self.loss_plot = MplCanvas_timing(
            self.ui.label_loss_plot, color=['b'], label=['loss'])
        self.update_plot = MplCanvas_timing(
            self.ui.label_update_plot, color=['b', 'k'], label=['using ML', 'no ML'])

    def run_Ackley(self, callback):
        x_train = []
        y_train = []

        # 開啟計時器
        self.start_time_counter()
        # 參數
        bounds = self.setting.params['bounds']
        popsize = self.setting.params['population size']
        generations = self.setting.params['generations']

        target_IQM = np.array([0])

        dimensions = self.setting.params['dimensions']
        param_value = np.array(self.setting.params['param_value'])  # 參數值
        param_change_idx = self.setting.params['param_change_idx']  # 有哪些參數需要更動
        param_change_num = len(param_change_idx)
        # print(bounds)

        # F = self.setting.params['F']
        # Cr = self.setting.params['Cr']
        # F_decay = self.setting.params['F_decay']
        # Cr_decay = self.setting.params['Cr_decay']

        # F_optimiter = HyperOptimizer(
        #     init_value=0.3, final_value=0.8, method="step", decay_value=-0.01)
        Cr_optimiter = HyperOptimizer(
            init_value=0.3, final_value=0.9, method="exponantial_reverse", rate=0.05)
        # Cr_optimiter = HyperOptimizer(
        #     init_value=0.5, final_value=0.5, method="constant")
        F_optimiter = HyperOptimizer(
            init_value=0.7, final_value=0.7, method="constant")
        # F_optimiter = HyperOptimizer(
        #     init_value=0.8, final_value=0.4, method="exponantial", rate=0.2)

        ##### ML #####
        self.ML = ML(self.setting.params['pretrain_model'], self.setting.params['train'], dimensions, 1)
        ##### ML #####

        # 初始化20群(population) normalize: [0, 1]
        pop = np.random.rand(popsize, param_change_num)
        pop = np.round(pop, 8)
        # pop = np.array([[0.5]*4] * popsize)

        # 取得每個參數的邊界
        min_b, max_b = np.asarray(bounds).T
        min_b = min_b[param_change_idx]
        max_b = max_b[param_change_idx]
        diff = np.fabs(min_b - max_b)

        #denormalize [min_b, max_b]
        pop_denorm = min_b + pop * diff

        ans = np.zeros(len(param_value))
        # ans[param_change_idx] = min_b + \
        #     np.around(np.random.rand(param_change_num), 4) * diff
        self.ui.label_ans.setText(str(ans[param_change_idx]))
        self.reset(popsize, param_change_num, ans[param_change_idx])
        # self.show_param_window_signal.emit()

        # measure score
        fitness = []
        IQMs = []
        for i in range(popsize):
            # replace fix value
            param_value[param_change_idx] = pop_denorm[i]
            f = self.fobj(param_value - ans)
            fitness.append(f)
            IQMs.append([f])
            self.update_param_window_signal.emit(i, pop_denorm[i], f, np.array([]))

        # find the best pop(以這個例子是score最小的pop)
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]

        param_value[param_change_idx] = best
        # ParamModifyBlock_idx = 0
        # for P in self.ui.ParamModifyBlock:
        #     for E in P.lineEdits:
        #         E.setText(str(np.around(param_value[ParamModifyBlock_idx], 4)))
        #         ParamModifyBlock_idx += 1

        self.ui.label_score.setText(str(np.round(fitness[best_idx], 5)))

        update_rate = 0
        ML_update_rate = 0
        # Do Differential Evolution
        for i in range(generations):
            update_times = 0
            ML_update_times = 0

            self.ui.label_generation.setText(str(i))
            F = F_optimiter.update(i)
            Cr = Cr_optimiter.update(i)

            for j in range(popsize):
                # sleep(1)
                self.ui.label_individual.setText(str(j))

                # select all pop except j
                idxs = [idx for idx in range(popsize) if idx != j]
                # random select two pop except j
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # Mutation
                mutant = np.clip(a + F*(b - c), 0, 1)

                # random choose the dimensions
                cross_points = np.random.rand(param_change_num) < Cr
                # if no dimensions be selected
                if not np.any(cross_points):
                    # random choose one dimensions
                    cross_points[np.random.randint(0, param_change_num)] = True

                # 隨機替換突變
                trial = np.where(cross_points, mutant, pop[j])
                trial = np.round(trial, 8)

                # denormalize
                trial_denorm = min_b + trial * diff

                # add fix value
                param_value[param_change_idx] = trial_denorm

                # mesure score
                f = self.fobj(param_value - ans)

                if f < fitness[j]: update_times += 1

                # use model to predict
                if self.ML.PRETRAIN_MODEL or (self.ML.TRAIN and i>=2):
                    diff_target_IQM = target_IQM - IQMs[j]
                    times = 0
                    x = np.zeros(dimensions)
                    x[param_change_idx] = trial - pop[j]
                    pred = self.ML.model(torch.FloatTensor([x.tolist()])).detach().numpy()
                    while (pred * diff_target_IQM <= 0).all() and times<20: # 如果預測分數會上升就重找參數
                        times+=1
                        # select all pop except j
                        idxs = [idx for idx in range(popsize) if idx != j]
                        
                        # random select two pop except j
                        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                        # Mutation
                        mutant = np.clip(a + F*(b - c), 0, 1)

                        # random choose the dimensions
                        cross_points = np.random.rand(param_change_num) < Cr
                        # if no dimensions be selected
                        if not np.any(cross_points):
                            # random choose one dimensions
                            cross_points[np.random.randint(0, param_change_num)] = True

                        # 隨機替換突變
                        trial = np.where(cross_points, mutant, pop[j])
                        trial = np.round(trial, 8)
                        x = np.zeros(dimensions)
                        x[param_change_idx] = trial - pop[j]
                        pred = self.ML.model(torch.FloatTensor([x.tolist()])).item()
                        
                print(times)
                # denormalize
                trial_denorm = min_b + trial * diff

                # add fix value
                param_value[param_change_idx] = trial_denorm

                ##### ML #####
                if self.ML.TRAIN and i>0:
                    # 建立一個子執行緒
                    self.train_task = threading.Thread(target = lambda: self.ML.train(i, x_train, y_train, self.loss_plot))
                    # self.ML.train(i, x_train, y_train, self.loss_plot)
                    # 當主程序退出，該執行緒也會跟著結束
                    self.train_task.daemon = True
                    # 執行該子執行緒
                    self.train_task.start()
                ##### ML #####

                # mesure score
                f = self.fobj(param_value - ans)
                if self.ML.TRAIN and i>0:
                    self.train_task.join()

                ##### ML #####
                if self.ML.PRETRAIN_MODEL or self.ML.TRAIN:
                    x = np.zeros(dimensions)
                    x[param_change_idx] = trial - pop[j]
                    y = [f-fitness[j]]
                    if f < fitness[j]: ML_update_times += 1
                    # else:
                    #     x_train.append(x.tolist())
                    #     y_train.append(y)

                    x_train.append(x.tolist())
                    y_train.append(y)
                ##### ML #####

                # 如果突變種比原本的更好
                if f < fitness[j]:
                    # 替換原本的個體
                    fitness[j] = f
                    pop[j] = trial
                    self.update_param_window_signal.emit(j, trial_denorm, f, np.array([]))

                    # 如果突變種比最優種更好
                    if f < fitness[best_idx]:
                        # 替換最優種
                        best_idx = j

                        # ParamModifyBlock_idx = 0
                        # for P in self.ui.ParamModifyBlock:
                        #     for E in P.lineEdits:
                        #         E.setText(str(np.around(param_value[ParamModifyBlock_idx], 4)))
                        #         ParamModifyBlock_idx += 1
                       
                # print(fitness[best_idx])
                self.ui.label_score.setText(str(np.round(fitness[best_idx], 5)))
                if f <= 1e-5: self.is_run = False

                self.bset_score_plot.update([fitness[best_idx]])
                self.hyper_param_plot.update([F, Cr])
                self.update_plot.update([ML_update_rate, update_rate])

                if not self.is_run:
                    callback()
                    sys.exit()
                    
            update_rate = update_times/popsize
            ML_update_rate = ML_update_times/popsize
        callback()

    def set_time_counter(self):
        time_counter = 0
        while self.is_run:
            total_sec = time_counter
            hour = max(0, total_sec//3600)
            minute = max(0, total_sec//60 - hour * 60)
            sec = max(0, (total_sec - (hour * 3600) - (minute * 60)))
            # show time_counter (by format)
            self.ui.label_time.setText(str(f"{hour}:{minute:0>2}:{sec:0>2}"))

            sleep(1)
            time_counter += 1

    def start_time_counter(self):
        # 建立一個子執行緒
        self.timer = threading.Thread(target=self.set_time_counter)
        # 當主程序退出，該執行緒也會跟著結束
        self.timer.daemon = True
        # 執行該子執行緒
        self.timer.start()

    def reset(self, popsize, param_change_num, ans):
        self.setup_param_window_signal.emit(popsize, param_change_num, ans)
        # reset plot
        self.bset_score_plot.reset()
        self.hyper_param_plot.reset()
        self.loss_plot.reset()
        self.update_plot.reset()
        self.ui.label_generation.setText("#")
        self.ui.label_individual.setText("#")
        self.ui.label_score.setText("#")

    # Ackley
    # objective function
    def fobj(self, X):
        firstSum = 0.0
        secondSum = 0.0
        for c in X:
            firstSum += c**2.0
            secondSum += math.cos(2.0*math.pi*c)
        n = float(len(X))
        return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
