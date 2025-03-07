import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
output_dir = "cropped"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all images from VIT_Dataset
dataset_dir = "VIT_Dataset"
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Load the image
    image_path = os.path.join(dataset_dir, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image: {image_file}")
        continue
    
    # Get filename without extension for saving crops
    base_name = os.path.splitext(image_file)[0]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Extract blue regions (stamps are usually blue or cyan)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 50, 50])  # Adjust based on the stamp color range
    upper_blue = np.array([140, 255, 255])
    stamp_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in stamp mask
    stamp_contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding rectangles
    stamp_regions = []
    signature_regions = []

    # Process stamp contours
    for i, contour in enumerate(stamp_contours):
        if cv2.contourArea(contour) > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            stamp = image[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f'{base_name}_stamp_{i}.png')
            cv2.imwrite(output_path, stamp)

    # Process signature contours
    for i, contour in enumerate(signature_contours):
        if cv2.contourArea(contour) > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            if not any(cv2.countNonZero(stamp_mask[y:y+h, x:x+w]) > 0.5 * w * h for stamp in stamp_regions):
                signature = image[y:y+h, x:x+w]
                output_path = os.path.join(output_dir, f'{base_name}_signature_{i}.png')
                cv2.imwrite(output_path, signature)

    print(f"Processed {image_file}")

print("All images processed successfully!")