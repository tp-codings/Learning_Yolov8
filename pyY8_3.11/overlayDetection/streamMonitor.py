import cv2
import screeninfo
import pyautogui
import numpy as np

monitors = screeninfo.get_monitors()

if not monitors:
    print("Keine Monitore gefunden.")
    exit()

selected_monitor = monitors[1]

monitor_x, monitor_y, monitor_width, monitor_height = selected_monitor.x, selected_monitor.y, selected_monitor.width, selected_monitor.height

cv2.namedWindow("Selected Monitor Screen", cv2.WINDOW_NORMAL)

while True:
    screenshot = np.array(pyautogui.screenshot(region=(monitor_x, monitor_y, monitor_width, monitor_height)))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    cv2.imshow("Selected Monitor Screen", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
