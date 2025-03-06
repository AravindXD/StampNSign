import os
import cv2
from ultralytics import YOLO
import numpy as np
import torch

# Load the trained model
model = YOLO("signature-detection-best.pt")

# Create output folders
output_folder = "signature_crops"
debug_folder = "debug_output"  # For visualization
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

def process_single_image(image_path, output_folder, debug_folder):
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return

        # Run inference with lower confidence threshold
        with torch.no_grad():
            results = model.predict(image_path, conf=0.1)  # Lowered from 0.25 to 0.1
            
            # Debug information
            print(f"Processing {image_path}")
            print(f"Number of detections: {len(results[0].boxes)}")
            print(f"Confidence scores: {results[0].boxes.conf if len(results[0].boxes) > 0 else 'None'}")
            
            # Create debug visualization
            debug_image = image.copy()
            
            if len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    conf = float(results[0].boxes.conf[i])
                    
                    # Draw rectangle on debug image
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug_image, f"conf: {conf:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save crop
                    crop = image[y1:y2, x1:x2].copy()
                    output_path = os.path.join(output_folder, 
                                             f"{os.path.basename(image_path)}_signature_{i + 1}.jpg")
                    cv2.imwrite(output_path, crop)
                    print(f"Saved crop: {output_path}")
                    del crop
            
            # Save debug visualization
            debug_path = os.path.join(debug_folder, f"debug_{os.path.basename(image_path)}")
            cv2.imwrite(debug_path, debug_image)
            print(f"Saved debug image: {debug_path}")

        del results
        del image
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# Process images one by one
input_folder = "VIT_Dataset"
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    process_single_image(image_path, output_folder, debug_folder)
    torch.cuda.empty_cache()
