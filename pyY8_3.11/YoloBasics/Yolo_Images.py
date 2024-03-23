from ultralytics import YOLO
import cv2
import torch

print(torch.cuda.is_available())

model = YOLO('YoloWeights/TD.pt')

results = model("Images\GZ8.jpeg", show=True)
cv2.waitKey(0)