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
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))  # Adjust extension if needed

for img_path in image_paths:
    # Load image and run inference
    results = model(img_path)

    # Read original image
    img = cv2.imread(img_path)

    for result in results:
        for box in result.boxes.xyxy:  # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Crop the detected bounding box region
            cropped_img = img[y1:y2, x1:x2]

            # Convert to HSV
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

            # Apply mask for blue color
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Apply the mask to the cropped image
            result_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

            # Save the result image
            filename = os.path.basename(img_path).replace(".jpg", f"_processed.jpg")
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, result_img)

print("Processing complete! Saved outputs in:", output_dir)