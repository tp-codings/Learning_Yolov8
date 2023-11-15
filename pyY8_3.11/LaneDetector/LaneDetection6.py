import cv2
import numpy as np
from LaneDetector import hermiteTest
vidcap = cv2.VideoCapture("Videos/LaneVideo.mp4")
success, image = vidcap.read()
import time

width = 640
height = 480

oldTime = 0
fpsFILT = 20

fac = 60
acc = 20
lastRange = 50

maskLineTop = 370
maskLineBottom = 470


def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

## Choosing points for perspective transformation
tl = (240, maskLineTop)
bl = (90 ,maskLineBottom)
tr = (400, maskLineTop)
br = (538, maskLineBottom)

carCenter = (width//2, maskLineBottom)

leftBottomLast = (0, 0)
leftTopLast = (width, height)

rightBottomLast = (0, 0)
rightTopLast = (width, height)

while success:
    newTime = time.time()
    deltaTime = newTime - oldTime
    oldTime = newTime
    fps = 1/deltaTime

    fpsFILT = fpsFILT*.95+fps*.05

    success, image = vidcap.read()

    frame = cv2.resize(image, (width, height))
    
    cv2.putText(frame, str(int(fpsFILT)), (5, 14), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)


    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    # Um die inverse Matrix zu erhalten
    inverse_matrix = np.linalg.inv(matrix)

    # Jetzt können Sie 'inverse_matrix' verwenden, um das Bild von der Vogelperspektive auf die Originalansicht zurückzutransformieren
    transformed_back_frame = cv2.warpPerspective(transformed_frame, inverse_matrix, (width, height))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    y = maskLineBottom
    lx = []
    rx = []
    msk = mask.copy()

    while y > 0:
        ## Left threshold
        img_left = mask[y - acc:y, left_base - fac:left_base + fac]
        contours_left, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ## Right threshold
        img_right = mask[y - acc:y, right_base - fac:right_base + fac]
        contours_right, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_left:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append((left_base - fac + cx, y))
                left_base = left_base - fac + cx


        for contour in contours_right:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                rx.append((right_base - fac + cx, y))
                right_base = right_base - fac + cx

        cv2.rectangle(msk, (left_base - fac, y), (left_base + fac, y - acc), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - fac, y), (right_base + fac, y - acc), (255, 255, 255), 2)
        y -= acc
    
    original_lx = []
    
    for i in range(len(lx)):
        x, y = lx[i][0], lx[i][1]
        if y > height-lastRange:
            leftBottomLast = (x, y)
        if y < lastRange:
            leftTopLast = (x, y)

        original_point = np.dot(inverse_matrix, np.array([x, y, 1]))
        original_lx.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))

    #BottomLeft
    if leftBottomLast[1] > 0:
        original_point = np.dot(inverse_matrix, np.array([leftBottomLast[0], leftBottomLast[1], 1]))
        original_lx.insert(0, (int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))
    
    #TopLeft
    if leftTopLast[1] < height:
        original_point = np.dot(inverse_matrix, np.array([leftTopLast[0], leftTopLast[1], 1]))
        original_lx.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))

    original_rx = []
    for i in range(len(rx)):
        x, y = rx[i][0], rx[i][1]
        if y > height-lastRange:
            rightBottomLast = (x, y)
        if y < lastRange:
            rightTopLast = (x, y) 
                       
        original_point = np.dot(inverse_matrix, np.array([x, y, 1]))
        original_rx.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))
        
    #BottomRight
    if rightBottomLast[1] > 0:
        original_point = np.dot(inverse_matrix, np.array([rightBottomLast[0], rightBottomLast[1], 1]))
        original_rx.insert(0, (int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))
    
    
    #TopRight
    if rightTopLast[1] < height:
        original_point = np.dot(inverse_matrix, np.array([rightTopLast[0], rightTopLast[1], 1]))
        original_rx.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))


    #cv2.imshow("Original Frame mit Punkten", frame)

    # for point in original_lx:
    #     cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # for point in original_rx:
    #     cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Sortiere die Punkte nach ihrer x-Koordinate
    original_lx = sorted(original_lx, key=lambda x: x[1])
    original_rx = sorted(original_rx, key=lambda x: x[1])

    # Berechne die Distanzen
    distLeft = carCenter[0] - original_lx[-1][0]
    distRight = original_rx[-1][0] - carCenter[0]

    print(distLeft, distRight)

    # Berechne die Spurbreite
    laneWidth = distLeft + distRight
    normFactor = (width // 3) / laneWidth

    lineHeight = 10

    # Berechne die Position des Verfolgers auf der Mittellinie
    centerTracker = (width // 3 + int(distLeft * normFactor), height // 3)

    # Zeichne die Linien und den Verfolger
    cv2.line(frame, (width // 3, height // 3), (centerTracker[0], height // 3), (0, 0, 0), 2)
    cv2.line(frame, (width // 3 * 2, height // 3), (centerTracker[0], height // 3), (0, 0, 0), 2)

    cv2.line(frame, (width // 3, height // 3 + lineHeight), (width // 3, height // 3 - lineHeight), (0, 0, 0), 2)
    cv2.line(frame, (width // 3 * 2, height // 3 + lineHeight), (width // 3 * 2, height // 3 - lineHeight), (0, 0, 0), 2)
    cv2.line(frame, (width // 2, height // 3 + lineHeight), (width // 2, height // 3 - lineHeight), (0, 255, 0), 2)

    cv2.circle(frame, centerTracker, 5, (0, 255, 0), -1)

    # Berechne die Abweichung des centerTracker vom tatsächlichen Mittelpunkt in Prozent
    actualCenter = width // 2
    distanceFromCenter = abs(centerTracker[0] - actualCenter)
    percentageDeviation = (distanceFromCenter / (width // 6)) * 100

    # Gebe den Prozentsatz der Abweichung aus
    # Wähle den Text basierend auf der größeren Distanz
    if distLeft > distRight:
        cv2.putText(frame, "{:.1f}%".format(percentageDeviation), (width // 2 + 20, height // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "{:.1f}%".format(percentageDeviation), (width // 3 + 20, height // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    # Konvertiere die Punkte in ein NumPy-Array
    original_lx = np.array(original_lx)
    original_rx = np.array(original_rx)
    if original_lx.any() and original_rx.any():
        contour_points = np.vstack((original_lx, original_rx[::-1]))   

        # Erstelle eine transparente Overlay-Schicht
        overlay = frame.copy()
        if contour_points.any():
            cv2.drawContours(overlay, [contour_points], -1, (0, 255, 0), thickness=cv2.FILLED)

        # Setze den Alpha-Wert für die Overlay-Schicht
        alpha = 0.5  # Hier kannst du den Alpha-Wert einstellen (0 für transparent, 1 für undurchsichtig)

        # Führe die Alpha-Blendung durch
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    #hermiteTest.interpolate(original_lx, 0.5, frame, height)
    #hermiteTest.interpolate(original_rx, 0.5, frame, height)

    if len(original_lx) > 1:
        pts_left = np.array(original_lx, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_left], isClosed=False, color=(255, 0, 0), thickness=2)

    if len(original_rx) > 1:
        pts_right = np.array(original_rx, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_right], isClosed=False, color=(0, 0, 255), thickness=2)




    cv2.imshow("Lane Detection", frame)
    
    #cv2.imshow("Original", frame)
    #cv2.imshow("Bird's Eye View", transformed_frame)
    #cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    #cv2.waitKey(0)
    if cv2.waitKey(10) == 27:
        break

