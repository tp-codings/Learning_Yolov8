from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import math

cap = cv2.VideoCapture("Videos/escalator.mp4")

model = YOLO("YoloWeights/yolov8l.pt")

classNames = model.names

mask = cv2.imread("Images/escalatorMask2.png")

#Tracking 
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsDown = [607, 800, 991, 800]
limitsUp = [1442, 576, 1917, 576]
totalCount = 0
countDown =  []
countUp = []


stop = False

while not stop:
    success, frame = cap.read()

    imgRegion = frame & mask

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))


    cvzone.putTextRect(frame, f"Down", (640, 830), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Up", (1850, 600), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)

    
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

            if currentClass == "person" and conf > 0.3:
                cvzone.putTextRect(frame, f"{conf} {currentClass}", (max(0, x1), max(50, y1)), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
                cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 0),  l = 5, t = 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)

    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3],), (0,0, 255), 2)
    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3],), (0,0, 255), 2)

    for r in resultsTracker:
        x1, y1, x2, y2, Id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        #cvzone.putTextRect(frame, f"{int(Id)}", (cx, cy), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)


        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-10 < cy < limitsDown[1] + 10 and Id not in countDown:
            countDown.append(Id)
            cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3],), (0,255, 0), 2)
        
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-10 < cy < limitsUp[1] + 10 and Id not in countUp:
            countUp.append(Id)
            cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3],), (0,255, 0), 2)

    lencountDown = len(countDown)
    lencountUp = len(countUp)
    totalCount = lencountDown + lencountUp
    cvzone.putTextRect(frame, f"Count: {totalCount}", (15, 50), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Up: {lencountDown}", (15, 90), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)
    cvzone.putTextRect(frame, f"Down: {lencountUp}", (15, 130), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, offset=5)

    cv2.imshow("Webcam", frame)
    #cv2.waitKey(0) 
    if cv2.waitKey(1) & 0xff == ord('q'):
        stop = True