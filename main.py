from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import f_Face_info
import cv2
import time
import imutils
from statistics import mean

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._detection_flag = False
        self._anonymize_flag = False

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            #start_time = time.time()
            ret, cv_img = cap.read()
            detection_result = {}
            frame = imutils.resize(cv_img, width=720)
            if self._detection_flag:
                # obtenego info del frame
                out = f_Face_info.get_face_info(frame)
                # pintar imagen
                res_img = f_Face_info.bounding_box(out, frame, self._anonymize_flag)

                #end_time = time.time() - start_time
                #FPS = 1 / end_time
                #cv2.putText(res_img, f"FPS: {round(FPS, 3)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                detection_result['info'] = self.personnelStatistics(out)
                detection_result['img'] = res_img
            else:
                detection_result['img'] = frame
            if ret:
                self.change_pixmap_signal.emit(detection_result)
        # shut down capture system
        cap.release()

    def toggleDetection(self, isOn):
        self._detection_flag = isOn

    def toggleAnonymize(self, isOn):
        self._anonymize_flag = isOn

    def personnelStatistics(self, detection):
        stats = {
            'peopleCount': len(detection),
            'maleCount': len([p for p in detection if p["gender"] == "Man"]),
            'femaleCount': len([p for p in detection if p["gender"] == "Woman"]),
            'avgAge': 0
        }
        ages = [float(p["age"]) for p in detection if len(p["age"]) > 0]
        if len(ages) > 0:
             stats['avgAge'] = mean(ages)
        return stats

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face recognition demo")
        self.display_width = 720
        self.display_height = 480

        # create the frame that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        # create start button
        self.startButton = QPushButton("Start AI", self)
        self.startButton.setGeometry(200, 150, 200, 50)
        self.startButton.setCheckable(True)
        self.startButton.clicked.connect(self.toggleDetection)
        self.startButton.setStyleSheet("background-color: teal; font-size: 15; color: white")

        # create start button
        self.anonymizeButton = QPushButton("Start anonymize", self)
        self.anonymizeButton.setGeometry(200, 150, 500, 50)
        self.anonymizeButton.setCheckable(True)
        self.anonymizeButton.clicked.connect(self.toggleAnonymize)
        self.anonymizeButton.setStyleSheet("background-color: teal; font-size: 15; color: white")

        # create button layout
        buttonWidget = QWidget()
        buttonLayout = QVBoxLayout(buttonWidget)
        buttonLayout.addWidget(self.startButton)
        buttonLayout.addWidget(self.anonymizeButton)

        # Create detection result display
        self.peopleLabel = QLabel('Number of persons: 0', self)
        self.peopleLabel.setStyleSheet("border: 1px solid; background-color: white")
        self.femaleLabel = QLabel('Female: 0', self)
        self.femaleLabel.setStyleSheet("border: 1px solid; background-color: white")
        self.maleLabel = QLabel('Male: 0', self)
        self.maleLabel.setStyleSheet("border: 1px solid; background-color: white")
        self.ageLabel = QLabel('Avg Age: 0', self)
        self.ageLabel.setStyleSheet("border: 1px solid; background-color: white")

        # Create text layout
        textWidget = QWidget()
        textLayout = QHBoxLayout(textWidget)
        textLayout.addWidget(self.peopleLabel)
        textLayout.addWidget(self.femaleLabel)
        textLayout.addWidget(self.maleLabel)
        textLayout.addWidget(self.ageLabel)

        # set layout
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addWidget(buttonWidget)
        vbox.addLayout(hbox)
        vbox.addWidget(textWidget)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def toggleDetection(self):
        if self.startButton.isChecked():
            self.startButton.setText("Stop AI")
            self.startButton.setStyleSheet("background-color: #016064; font-size: 15; color: white")
        else:
            self.startButton.setText("Start AI")
            self.startButton.setStyleSheet("background-color: teal; font-size: 15; color: white")
        if self.thread is not None:
            self.thread.toggleDetection(self.startButton.isChecked())

    def toggleAnonymize(self):
        if self.anonymizeButton.isChecked():
            self.anonymizeButton.setText("Stop anonymize")
            self.anonymizeButton.setStyleSheet("background-color: #016064; font-size: 15; color: white")
        else:
            self.anonymizeButton.setText("Start anonymize")
            self.anonymizeButton.setStyleSheet("background-color: teal; font-size: 15; color: white")
        if self.thread is not None:
            self.thread.toggleAnonymize(self.anonymizeButton.isChecked())

    @pyqtSlot(object)
    def update_image(self, result):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(result['img'])
        self.image_label.setPixmap(qt_img)
        if 'info' in result:
            self.update_text(result['info'])
        else:
            self.update_text()

    def update_text(self, info=None):
        if info is not None:
            self.peopleLabel.setText(f'Number of persons: {info["peopleCount"]}')
            self.maleLabel.setText(f'Male: {info["maleCount"]}')
            self.femaleLabel.setText(f'Female: {info["femaleCount"]}')
            self.ageLabel.setText(f'Avg Age: {str(round(info["avgAge"], 2))}')
        else:
            self.peopleLabel.setText(f'Number of persons: 0')
            self.maleLabel.setText(f'Male: 0')
            self.femaleLabel.setText(f'Female: 0')
            self.ageLabel.setText(f'Avg Age: 0')

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())