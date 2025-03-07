import cv2
import os
import glob
import numpy as np
import torch
from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Input and output directories
input_dir = "VIT_Dataset"
output_dir = "final_output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define HSV range for blue color
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Get all image paths from the dataset directory
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

for img_path in image_paths:
    # Load image and run inference
    results = model(img_path)
    img = cv2.imread(img_path)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = img[y1:y2, x1:x2]

            # Convert to HSV and create mask
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Create transparent background
            bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = np.where(mask == 255, 255, 0)

            # Save as PNG with transparency
            filename = os.path.basename(img_path).replace(".jpg", "_processed.png")
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, bgra)

print("Processing complete! Transparent stamps saved in:", output_dir)
