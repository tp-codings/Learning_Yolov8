import cv2
import numpy as np

vidcap = cv2.VideoCapture("LaneVideo.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while success:
    success, image = vidcap.read()
    frame = cv2.resize(image, (640, 480))

    ## Choosing points for perspective transformation
    tl = (222,387)
    bl = (70 ,472)
    tr = (400,380)
    br = (538,472)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))
    # Um die inverse Matrix zu erhalten
    inverse_matrix = np.linalg.inv(matrix)

    # Jetzt können Sie 'inverse_matrix' verwenden, um das Bild von der Vogelperspektive auf die Originalansicht zurückzutransformieren
    transformed_back_frame = cv2.warpPerspective(transformed_frame, inverse_matrix, (640, 480))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    center_base = (left_base + right_base) // 2

    #Sliding Window
    y = 472
    lx = []
    rx = []
    center_line = []

    msk = mask.copy()

    while y > 0:
        ## Left threshold
        img_left = mask[y - 40:y, left_base - 50:left_base + 50]
        contours_left, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_left:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        ## Right threshold
        img_right = mask[y - 40:y, right_base - 50:right_base + 50]
        contours_right, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_right:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        ## Center threshold
        img_center = mask[y - 40:y, center_base - 25:center_base + 25]
        contours_center, _ = cv2.findContours(img_center, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_center:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_global = center_base - 25 + cx
                center_line.append(cx_global)

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (center_base - 25, y), (center_base + 25, y - 40), (255, 255, 255), 2)

        y -= 40
    
    original_lx = []
    for i in range(len(lx)):
        x, y = lx[i], 472 - i * 40  
        original_point = np.dot(inverse_matrix, np.array([x, y, 1]))
        original_lx.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))

    original_rx = []
    for i in range(len(rx)):
        x, y = rx[i], 472 - i * 40  
        original_point = np.dot(inverse_matrix, np.array([x, y, 1]))
        original_rx.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))

    original_center = []
    for i in range(len(center_line)):
        x, y = center_line[i], 472 - i * 40  
        original_point = np.dot(inverse_matrix, np.array([x, y, 1]))
        original_center.append((int(original_point[0] / original_point[2]), int(original_point[1] / original_point[2])))

    # Extrahieren Sie x- und y-Koordinaten der linken und rechten Punkte
    lx_x, lx_y = zip(*original_lx) if len(original_lx) > 0 else ([], [])
    rx_x, rx_y = zip(*original_rx) if len(original_rx) > 0 else ([], [])
    center_x, center_y = zip(*original_center) if len(original_center) > 0 else ([], [])

    print("------------")
    print(lx_x, lx_y)
    print(rx_x, rx_y)
    print(center_x, center_y)

    # Interpolation mit einem Polynom des Grad 2 (Sie können den Grad anpassen)
    degree = 3

    # Überprüfen Sie, ob genügend Punkte für die Interpolation vorhanden sind
    if len(lx_x) > 2 and len(lx_y) > 2:
        lx_poly = np.polyfit(lx_y, lx_x, degree)
        interpolated_y = np.linspace(min(lx_y), max(lx_y), 100)
        interpolated_lx = np.polyval(lx_poly, interpolated_y)

    if len(rx_x) > 2 and len(rx_y) > 2:
        rx_poly = np.polyfit(rx_y, rx_x, degree)
        interpolated_y = np.linspace(min(rx_y), max(rx_y), 100)
        interpolated_rx = np.polyval(rx_poly, interpolated_y)

    if len(center_x) > 2 and len(center_y) > 2:
        center_poly = np.polyfit(center_y, center_x, degree)
        interpolated_y = np.linspace(min(center_y), max(center_y), 100)
        interpolated_center = np.polyval(center_poly, interpolated_y)

    # Zeichnen Sie die Originalpunkte auf dem Bild
    for point in original_lx:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    for point in original_rx:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    for point in original_center:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Konvertieren Sie die interpolierten Punkte in das Format für die Verwendung in cv2.polylines
    if len(lx_x) > 2 and len(lx_y) > 2:
        pts_left = np.array(list(zip(interpolated_lx, interpolated_y)), np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_left], isClosed=False, color=(255, 0, 0), thickness=2)

    if len(rx_x) > 2 and len(rx_y) > 2:
        pts_right = np.array(list(zip(interpolated_rx, interpolated_y)), np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_right], isClosed=False, color=(0, 0, 255), thickness=2)

    if len(center_x) > 2 and len(center_y) > 2:
        pts_center = np.array(list(zip(interpolated_center, interpolated_y)), np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_center], isClosed=False, color=(0, 255, 0), thickness=2)

    # Anzeigen des Bildes mit den dargestellten Punkten und Kurven
    cv2.imshow("Original Frame mit Punkten und Kurven", frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    if cv2.waitKey(10) == 27:
        break
