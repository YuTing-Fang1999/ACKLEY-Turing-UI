import cv2
import os
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal

from time import sleep
from subprocess import check_output, call
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

    def capture(self, path = "", focus_time = 4, save_time = 1, num = 0):
        self.capture_fail_signal.emit()
        self.state.acquire()
        self.state.wait()
        self.state.release()

    def open_camera(self):
        os.system("adb shell am start -a android.media.action.STILL_IMAGE_CAMERA --ez com.google.assistant.extra.CAMERA_OPEN_ONLY true --ez android.intent.extra.CAMERA_OPEN_ONLY true --ez isVoiceQuery true --ez NoUiQuery true --es android.intent.extra.REFERRER_NAME android-app://com.google.android.googlequicksearchbox/https/www.google.com")

                
    def clear_camera_folder(self):
        #delete from phone: adb shell rm self.CAMERA_PATH/*
        if self.CAMERA_DEBUG: print('clear_camera_folder')
        r = check_output(['adb','shell','rm','-rf',self.CAMERA_PATH + '*'])
        if self.CAMERA_DEBUG: print(r.strip())

    def press_camera_button(self):
        #condition 1 screen on 2 camera open: adb shell input keyevent = CAMERA
        if self.CAMERA_DEBUG: print('press_camera_button')
        call(['adb','shell','input','keyevent = CAMERA'])
        
    def transfer_img(self, path='', num = 1):
        if self.CAMERA_DEBUG: print('screen transfer_img')
        # list all file
        r = check_output(['adb','shell','ls','-lt', self.CAMERA_PATH])
        if self.CAMERA_DEBUG: print('all files\n',r.decode('utf-8'))

        # find the last num
        file_names = r.decode('utf-8').split('\n')[1:num+1] 
        file_names = [f.split(' ')[-1][:-1] for f in file_names]
        # print(file_names)

        # pull the file and rename to path
        if path == '': path = file_name
        if file_names[0] == '':
            # print('拍攝未成功，請檢查後再次拍攝')
            # self.capture_fail()
            input('拍攝未成功，請檢查後按enter鍵再次拍攝')
            print('正在重新拍攝...')
            self.capture(path)
        else:
            for i in range(num):
                file_name = self.CAMERA_PATH + file_names[i]
                p = str(path+"_"+str(i)+".jpg")
                print('transfer',file_name,'to',p)
                r = check_output(['adb','pull', file_name, p])
                if self.CAMERA_DEBUG: print(r.strip())


    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def getROIposition(self, img):
        resize = self.ResizeWithAspectRatio(img, width=1280) # Resize by width OR
        scale = img.shape[0] / resize.shape[0]

        roi = cv2.selectROI(windowName="roi", img=resize, showCrosshair=False, fromCenter=False)
        roi = list(map(lambda x: int(x * scale), roi))
    #     x, y, w, h = roi

        # cv2.imshow("roi", img[y : y+h, x:x+w])
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        return roi

    def select_ROI(self):
        # capture 
        img_name = 'capture'
        self.capture(img_name, num=1)
        sleep(1)
        # # draw rectangle ROI
        img = cv2.imread(img_name+"_0.jpg")
        # self.set_img(img, self.ui.label_ROI_img)
        x, y, w, h = self.getROIposition(img)
        color = (0, 0, 255) # red
        thickness = 10 # 寬度 (-1 表示填滿)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
        # display img
        self.set_img(img, self.ui.label_ROI_img)
        return x, y, w, h

    def set_img(self, img, label):
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
        qimg = QPixmap(qimg)
        label.setPixmap(qimg)
