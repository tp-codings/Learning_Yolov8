import cv2
import screeninfo
import pyautogui
import numpy as np
from ultralytics import YOLO
import cvzone
import math

monitors = screeninfo.get_monitors()

if not monitors:
    print("Keine Monitore gefunden.")
    exit()

selected_monitor = monitors[1]

monitor_x, monitor_y, monitor_width, monitor_height = selected_monitor.x, selected_monitor.y, selected_monitor.width, selected_monitor.height

model = YOLO("YoloWeights/fighterDetection_y8l.pt.pt")

classNames = model.names

while True:
    screenshot = np.array(pyautogui.screenshot(region=(monitor_x, monitor_y, monitor_width, monitor_height)))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Boundingbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 0))

            # Confidence und Classnames
            conf = math.ceil((box.conf[0] * 100)) / 100

            cvzone.putTextRect(frame, f"{conf} {classNames[int(box.cls[0])]}", (max(0, x1 + 30), max(30, y1)),
                               colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)

    # Zeige das Bild als Overlay auf dem Monitor an
    cv2.imshow("Selected Monitor Screen", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()