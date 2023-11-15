from ultralytics import YOLO
import cv2
import cvzone
import math
from PokerHandDetection import pokerHandCalculator

img = cv2.imread('Images\pokerHand.png') 

model = YOLO("YoloWeights/playingCardDetection.pt")

classNames = model.names

results = model('Images\pokerHand.png')


while True:
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Boundingbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3) #cv2
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 0))

            #Confidence und Classnames
            conf = math.ceil((box.conf[0]*100))/100

            className = classNames[int(box.cls[0])]

            cvzone.putTextRect(img, f"{conf} {className}", (max(0, x1+30), max(30, y1)), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)

            if conf > 0.5:
                hand.append(className)

    hand = list(set(hand))

    if len(hand) == 5:        

        pokerHand = pokerHandCalculator.findPokerHand(hand)
        cvzone.putTextRect(img, f"{pokerHand}", (30, 30), colorR=(0, 0, 0), scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Poker Hand Detector", img)
    cv2.waitKey(0)