import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QBrush, QColor, QImage, QFont
from PyQt5.QtWidgets import QApplication, QWidget
from ultralytics import YOLO
import pyautogui
import screeninfo
import numpy as np
import math
import cvzone
import cv2



# Global variables for circle parameters
circle_radius = 50
outline_width = 2
outline_color = Qt.white

model = YOLO("YoloWeights/fighterDetection_y8l.pt")
classNames = model.names


class GameOverlay(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()

        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        monitors = screeninfo.get_monitors()

        if not monitors:
            print("Keine Monitore gefunden.")
            exit()

        selected_monitor = monitors[1]

        self.monitor_x, self.monitor_y, self.monitor_width, self.monitor_height = selected_monitor.x, selected_monitor.y, selected_monitor.width, selected_monitor.height


        self.setGeometry(0, 0, screen_width, screen_height)  # Set the overlay size to match the screen
        self.show()

    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw a transparent background
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        painter.drawRect(self.rect())

        painter.setPen(QColor(outline_color))
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))

        screenshot = np.array(pyautogui.screenshot(region=(self.monitor_x, self.monitor_y, self.monitor_width, self.monitor_height)))

        frame = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        results = model(frame, stream=True)
        for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Boundingbox
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    #cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 0))

                    # Confidence und Classnames
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    if conf > 0.4:
                        #cvzone.putTextRect(frame, f"{conf} {classNames[int(box.cls[0])]}", (max(0, x1 + 30), max(30, y1)),
                                        #colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                        
                        #img = self.numpy_array_to_qimage(frame)
                        #painter.drawImage(self.rect(), img)
                        font = QFont("Helvetica", 15)
                        painter.setFont(font)
                        painter.drawText(x1, y1,f"{conf} {classNames[int(box.cls[0])]}")

                        self.drawRectangle(painter, x1, y1, w, h, Qt.red)

    def numpy_array_to_qimage(self, img):
        im_np = np.array(img)    
        qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0],                                                                                                                                                 
                        QImage.Format_BGR888)

        return qimage


    def drawRectangle(self, painter, x, y, width, height, color):
        painter.setPen(QColor(color))
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        painter.drawRect(x, y, width, height)


def main():
    app = QApplication(sys.argv)
    overlay = GameOverlay()

    timer = QTimer()
    timer.timeout.connect(overlay.update)
    timer.start(32)  

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
