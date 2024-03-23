from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("Videos/GZ2.mp4")

model = YOLO("YoloWeights/Train_Detection.pt")

classNames = model.names


while True:
    success, frame = cap.read()

    results = model(frame, stream=True)

    
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Boundingbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3) #cv2
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 0))

            #Confidence und Classnames
            conf = math.ceil((box.conf[0]*100))/100

            cvzone.putTextRect(frame, f"{conf} {classNames[int(box.cls[0])]}", (max(0, x1+30), max(30, y1)), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)


    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)