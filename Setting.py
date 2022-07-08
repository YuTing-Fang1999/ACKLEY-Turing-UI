import json
import os
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.Qt import Qt

import xml.etree.ElementTree as ET


import xml.etree.ElementTree as ET


class Setting():
    def __init__(self, ui):
        self.ui = ui

        if os.path.exists('setting.json'):
            self.read_setting()
            self.set_UI()

        else:
            # defult
            # 參數
            self.params = {
                'F': 0.5,
                'Cr': 0.7,
                'population size': 15,
                'generations': 1000,
                'target_IQM': [8.88249107, 1.95913905],
                'weight_IQM': [0.5, 0.5],
                'param_names': ['noise_profile_y', 'noise_profile_cb', 'noise_profile_cr',\
                        'denoise_scale_y', 'denoise_scale_chroma',\
                        'denoise_edge_softness_y', 'denoise_edge_softness_chroma',\
                        'denoise_weight_y', 'denoise_weight_chroma'],

            }

    def set_dimensions_lengths_defult_range(self, param_names, xml_path):
        # 從檔案載入並解析 XML 資料
        if not os.path.exists(xml_path): 
            print('No such file or directory:', xml_path)
            return
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 子節點與屬性
        wnr24_rgn_data  =  root.find("chromatix_wnr24_core/mod_wnr24_post_scale_ratio_data/"\
                                        "post_scale_ratio_data/mod_wnr24_pre_scale_ratio_data/"\
                                        "pre_scale_ratio_data/mod_wnr24_total_scale_ratio_data/"\
                                        "total_scale_ratio_data/mod_wnr24_drc_gain_data/"\
                                        "drc_gain_data/mod_wnr24_hdr_aec_data/hdr_aec_data/"\
                                        "mod_wnr24_aec_data/wnr24_rgn_data"
                                        )

        self.params['dimensions'] = 0
        self.params['lengths'] = []
        self.params['defult_range'] = []
        self.params['param_value'] = []

        for param_name in param_names:
            parent = wnr24_rgn_data.find(param_name+'_tab')
            # print(param_name, length, param_value, bound)

            param_value = parent.find(param_name).text.split(' ')
            param_value = [np.around(float(x),4) for x in param_value]

            bound = json.loads(parent.attrib['range'])
            length = int(parent.attrib['length'])

            self.params['dimensions'] += length
            self.params['lengths'].append(length)
            self.params['defult_range'].append(bound)
            self.params['param_value'].append(param_value)

        # converting 2d list into 1d
        self.params['param_value'] = sum(self.params['param_value'], [])
        # print(self.params['param_value'])
        idx = 0
        for P in self.ui.ParamModifyBlock:
            for lineEdit in P.lineEdits:
                lineEdit.setText(str(self.params['param_value'][idx]))
                idx+=1

    def set_project(self, folder_path):
        self.params['project_path'] = folder_path
        self.params['project_name'] = self.params['project_path'].split('/')[-1]
        self.params['tuning_dir'] = '/'.join(self.params['project_path'].split('/')[:-1])
        self.params['xml_path'] = self.params['project_path']+'/Scenario.Default/XML/OPE/wnr24_ope.xml'
        # self.xml_path = 'wnr24_ope.xml'
        
        # print(self.params['tuning_dir'])
        # print(self.params['xml_path'])

        self.set_dimensions_lengths_defult_range(self.params['param_names'], self.params['xml_path'])
        self.set_UI()

            
    def read_setting(self):
        print("read setting")
        # Opening JSON file
        f = open('setting.json')
        # returns JSON object as a dictionary
        self.params = json.load(f)

        # ans
        # Opening JSON file
        f = open('ans.json')
        # returns JSON object as a dictionary
        self.ans = json.load(f)

    def write_setting(self):
        print('write_setting')
        self.set_param()
        self.params.pop('bounds', None)
        with open("setting.json", "w") as outfile:
            json.dump(self.params, outfile)
        

    def set_param(self):
        # self.params['bin_name'] = self.ui.lineEdits_bin_name.text()
        # param
        key = self.ui.hyper_param_title
        for i in range(len(key)):
            if i//2 == 0: self.params[key[i]] = float(self.ui.lineEdits_hyper_setting[i].text())
            if i//2 == 1: self.params[key[i]] = int(self.ui.lineEdits_hyper_setting[i].text())
            if i//2 == 2: self.params[key[i]] = float(self.ui.lineEdits_hyper_setting[i].text())
            # if i//2 == 2: self.params[key[i]] = json.loads(self.ui.lineEdits_hyper_setting[i].text())

        # range
        self.params['range'] = []
        for i in range(9):
            self.params['range'].append(json.loads(self.ui.lineEdits_range[i].text()))

        self.params['bounds'] = [self.params['range'][0]]*self.params['lengths'][0]
        for i in range(1, len(self.params['lengths'])):
            self.params['bounds'] = np.concatenate([self.params['bounds'] , [self.params['range'][i]]*self.params['lengths'][i]])

        # param fix
        param_change_idx = []
        param_value = []
        idx = 0
        for P in self.ui.ParamModifyBlock:
            # print(len(P.checkBoxes))
            for i in range(len(P.checkBoxes)):
                if P.checkBoxes[i].isChecked():
                    if P.lineEdits[i].text() == "":
                        print(P.title, "有參數打勾卻未填入數字")
                        return False
                else:
                    param_change_idx.append(idx)
                param_value.append(float(P.lineEdits[i].text()))
                idx+=1

        self.params['param_change_idx'] = param_change_idx
        self.params['param_value'] = param_value

        print("set param successfully!")

        return True


    def set_UI(self):

        # hyper param
        key = self.ui.hyper_param_title
        for i in range(len(key)):
            self.ui.lineEdits_hyper_setting[i].setText(str(self.params[key[i]]))

        # range
        for i in range(9):
            self.ui.label_defult_range[i].setText(str(self.params['defult_range'][i]))
            self.ui.lineEdits_range[i].setText(str(self.params['range'][i]))

        img_name = 'capture_0.jpg'
        if os.path.exists(img_name) and self.params['roi']:
            img = cv2.imread(img_name)
            x, y, w, h = self.params['roi']
            # draw rectangle ROI
            color = (0, 0, 255) # red
            thickness = 20 # 寬度 (-1 表示填滿)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
            # display img
            self.set_img(img, self.ui.label_ROI_img)

        # param fix
        idx = 0
        for P in self.ui.ParamModifyBlock:
            for i in range(len(P.checkBoxes)):
                if idx not in self.params['param_change_idx']:
                    P.checkBoxes[i].setChecked(True)
                # P.lineEdits[i].setText(str(self.params['param_value'][idx]))
                P.lineEdits[i].setText(str(self.ans['param_value'][idx]))
                idx+=1


    def set_img(self, img, label):
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap(qimg))

        
        