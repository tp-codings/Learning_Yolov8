from ultralytics import YOLO
import cv2
import torch

print(torch.cuda.is_available())

model = YOLO('YoloWeights/playingCardDetection.pt')

results = model("Images\Poker-hands-sheet.jpg", show=True)
cv2.waitKey(0)