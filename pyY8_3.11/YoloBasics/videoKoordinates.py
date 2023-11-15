import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Klick bei den Koordinaten ({x}, {y})")

cap = cv2.VideoCapture('Videos/escalator.mp4')  

cv2.namedWindow('Video')

cv2.setMouseCallback('Video', click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(0):
        break

cap.release()
cv2.destroyAllWindows()