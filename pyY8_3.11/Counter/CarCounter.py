from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import math

cap = cv2.VideoCapture("Videos/traffic2.mp4")

model = YOLO("YoloWeights/yolov8l.pt")

classNames = model.names

#maskLeft = cv2.imread("Images/trafficMask1.png")

#Tracking 
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsLeft = [10, 290, 610, 330]
limitsRight = [630, 330, 1250, 290]
totalCount = 0
countLeft =  []
countRight = []


stop = False

while not stop:
    success, frame = cap.read()

    results = model(frame, stream=True)

    detections = np.empty((0, 5))

    #imgRegionLeft = cv2.bitwise_and(frame, maskLeft)

    cvzone.putTextRect(frame, f"Left", (40, 330), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Rigth", (1160, 330), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)

    
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Boundingbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            #Confidence und Classnames
            conf = math.ceil((box.conf[0]*100))/100
            currentClass = classNames[int(box.cls[0])]

            if currentClass == "car" or currentClass == "truck" or currentClass == "motorbike" or currentClass == "bus" and conf > 0.3:
                cvzone.putTextRect(frame, f"{conf} {currentClass}", (max(0, x1), max(30, y1)), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
                cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 0),  l = 5, t = 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)

    cv2.line(frame, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3],), (0,0, 255), 2)
    cv2.line(frame, (limitsRight[0], limitsRight[1]), (limitsRight[2], limitsRight[3],), (0,0, 255), 2)

    for r in resultsTracker:
        x1, y1, x2, y2, Id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if limitsLeft[0] < cx < limitsLeft[2] and limitsLeft[1]-10 < cy < limitsLeft[1] + 10 and Id not in countLeft:
            countLeft.append(Id)
            cv2.line(frame, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3],), (0,255, 0), 2)
        
        if limitsRight[0] < cx < limitsRight[2] and limitsRight[1]-10 < cy < limitsRight[1] + 10 and Id not in countRight:
            countRight.append(Id)
            cv2.line(frame, (limitsRight[0], limitsRight[1]), (limitsRight[2], limitsRight[3],), (0,255, 0), 2)

    lenCountLeft = len(countLeft)
    lenCountRight = len(countRight)
    totalCount = lenCountLeft + lenCountRight
    cvzone.putTextRect(frame, f"Count: {totalCount}", (15, 50), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Left: {lenCountLeft}", (15, 90), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Right: {lenCountRight}", (15, 130), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        stop = True