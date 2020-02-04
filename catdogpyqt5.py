# -*- coding: utf-8 -*-
 
from PyQt5 import QtCore, QtGui, QtWidgets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
 
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
 
model = keras.models.load_model(
    r'ENTER PATH') # path to model
 
 
classification = ['Cat', 'Dog']
 
 
def predict_img(img_path):
    img = keras.preprocessing.image.load_img(
        img_path, target_size=IMAGE_SIZE)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
 
    prediction = model.predict(img_tensor)
    top = np.argmax(prediction)
    print(classification[top])
    return classification[top]
 
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(712, 506)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
       
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(20, 10, 471, 431))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap(r"ENTER PATH")) # path to first photo
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
       
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(527, 260, 171, 81))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.label.setFont(font)
        self.label.setObjectName("label")
       
        self.button1 = QtWidgets.QPushButton(self.centralwidget)
        self.button1.setGeometry(QtCore.QRect(510, 70, 181, 101))
        self.button1.setObjectName("button1")
        self.button1.clicked.connect(self.on_click_select_and_predict_photo)
       
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 712, 30))
        self.menubar.setObjectName("menubar")
       
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
 
    def on_click_select_and_predict_photo(self):
        fname = QFileDialog.getOpenFileName(self.button1, 'Open file',
         r'ENTER PATH',"Image files (*.jpg)") # linux r'/home', Windows r'C:\'
        path = fname[0]
        self.photo.setPixmap(QtGui.QPixmap(path))
        predict = predict_img(path)
        self.label.setText(predict)
 
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cat or Dog"))
        self.label.setText(_translate("MainWindow", "Result"))
        self.button1.setText(_translate("MainWindow", "Select photo and predict"))
 
 
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
