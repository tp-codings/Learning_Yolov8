from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import math

cap = cv2.VideoCapture("Videos/GZ2.mp4")

model = YOLO("YoloWeights/Train_Detection.pt")

classNames = model.names
colors = [
    (255, 0, 0),     # 'autorack' - Rot
    (0, 255, 0),     # 'boxcar' - Grün
    (0, 0, 255),     # 'cargo' - Blau
    (255, 255, 0),   # 'container' - Gelb
    (255, 165, 0),   # 'flatcar' - Orange
    (128, 0, 128),   # 'flatcar_bulkhead' - Violett
    (128, 128, 128), # 'gondola' - Grau
    (0, 255, 255),   # 'hopper' - Türkis
    (255, 192, 203), # 'locomotive' - Rosa
    (0, 128, 0),     # 'passenger' - Dunkelgrün
    (70, 130, 180)   # 'tank' - Stahlblau (Beispielwert für 'tank')
]

print(classNames)

#mask = cv2.imread("Images/Mask1.png")

#Tracking 
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

x = 700

width = 1400
height = 720

limits = [x, 0, x, 720]
counts =  []

new_train = True

train_counter = 0

zero_count = 0

stop = False

while not stop:
    success, frame = cap.read()
    frame = cv2.resize(frame, (width, height))

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)


    results = model(grayFrame, stream=True, verbose = False)

    detections = np.empty((0, 5))

    #imgRegion = cv2.bitwise_and(frame, mask)

    cvzone.putTextRect(frame, f"Trackline", (40, 330), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)

    
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            zero_count += 1
        else:
            zero_count = 0

        for box in boxes:

            #Boundingbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            #Confidence und Classnames
            conf = math.ceil((box.conf[0]*100))/100
            currentClass = classNames[int(box.cls[0])]
            currentColor = colors[int(box.cls[0])]

            if  conf > 0.4 and currentClass != "passenger":
                if currentClass == "locomotive" and new_train:
                    train_counter += 1
                    new_train = False
                cvzone.putTextRect(frame, f"{conf} {currentClass}", (max(0, x1), max(30, y1)), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
                cvzone.cornerRect(frame, (x1, y1, w, h), colorC= currentColor, colorR=(0, 0, 0),  l = 5, t = 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    if zero_count >= 100:
        new_train = True

    resultsTracker = tracker.update(detections)

    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3],), (0,0, 255), 2)

    for r in resultsTracker:
        x1, y1, x2, y2, Id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if limits[0] - 20 < cx < limits[2] + 20 and limits[1]-10 < cy < limits[3] + 10 and Id not in counts:
            counts.append(Id)
            cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3],), (0,255, 0), 2)
        

    lencounts = len(counts)

    cvzone.putTextRect(frame, f"Totalcount: {lencounts}", (15, 90), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"TrainCount: {train_counter}", (15, 110), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"new train: {new_train}", (15, 130), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Zero Counts since:  {zero_count}", (15, 150), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)

    cv2.imshow("TrainCounter", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        stop = True