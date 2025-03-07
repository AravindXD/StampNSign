import cv2
import numpy as np
import os

# Create base Preds directory
os.makedirs("Preds", exist_ok=True)

def process_image(image_path):
    # Get image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create directory structure
    pred_dir = os.path.join("Preds", image_name)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Load and process image
    image = cv2.imread(image_path)
    # Check if the image is loaded properly
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return  # Skip further processing
    
    original = image.copy()

    # Save original image for tracking
    cv2.imwrite(os.path.join(pred_dir, "01_original.jpg"), image)

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

    # Dilate stamp mask to ensure we capture the entire stamp
    kernel = np.ones((5,5), np.uint8)
    dilated_stamp_mask = cv2.dilate(stamp_mask, kernel, iterations=1)

    # Create a mask for signatures by removing stamp regions from binary image
    signature_mask = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(dilated_stamp_mask))

    # Find contours in masks
    stamp_contours, _ = cv2.findContours(dilated_stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_contours, _ = cv2.findContours(signature_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create visualization images
    stamp_vis = original.copy()
    
    # Process stamp contours
    stamp_count = 0
    for i, contour in enumerate(stamp_contours):
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            padding = 10
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(image.shape[1] - x_pad, w + 2*padding)
            h_pad = min(image.shape[0] - y_pad, h + 2*padding)
            
            # Extract stamp region
            stamp_region = original[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Convert to grayscale for finding content
            stamp_gray = cv2.cvtColor(stamp_region, cv2.COLOR_BGR2GRAY)
            _, stamp_thresh = cv2.threshold(stamp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find actual content boundaries
            coords = cv2.findNonZero(stamp_thresh)
            if coords is not None:
                x_min, y_min, w_tight, h_tight = cv2.boundingRect(coords)
                stamp_region = stamp_region[y_min:y_min+h_tight, x_min:x_min+w_tight]
                
                # Save tightly cropped stamp
                stamp_filename = os.path.join(pred_dir, f"Stamp{stamp_count}.jpg")
                cv2.imwrite(stamp_filename, stamp_region)
                stamp_count += 1
            
            cv2.rectangle(stamp_vis, (x_pad, y_pad), (x_pad+w_pad, y_pad+h_pad), (0, 255, 0), 3)

    # Process signature contours
    sign_count = 0
    for i, contour in enumerate(signature_contours):
        area = cv2.contourArea(contour)
        if 8000 < area < 15000:  # 4x stricter area thresholds
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this region overlaps with any blue regions
            roi_mask = np.zeros_like(stamp_mask)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            blue_overlap = cv2.bitwise_and(stamp_mask, roi_mask)
            blue_ratio = cv2.countNonZero(blue_overlap) / area
            
            if blue_ratio < 0.1:  # Only process if region is not predominantly blue
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                solidity = float(area)/hull_area if hull_area > 0 else 0
                extent = float(area) / (w * h)
                
                # 4x stricter geometric requirements
                if (1.9 < float(w)/h < 2.1 and  # Much stricter aspect ratio
                    300 < w < 400 and 100 < h < 130 and  # Tighter size limits
                    circularity < 0.25 and 
                    0.45 < solidity < 0.55 and
                    0.35 < extent < 0.40):
                    
                    roi = signature_mask[y:y+h, x:x+w]
                    
                    # Gradient analysis for stroke direction
                    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_angles = np.arctan2(sobely, sobelx) * 180 / np.pi
                    angle_hist = np.histogram(gradient_angles, bins=36)[0]
                    angle_variation = np.std(angle_hist) / np.mean(angle_hist) if np.mean(angle_hist) > 0 else 0
                    
                    # Component and stroke analysis with stricter requirements
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
                    if 6 < num_labels < 15:  # Stricter component count
                        comp_areas = stats[1:, cv2.CC_STAT_AREA]
                        comp_var = np.std(comp_areas) / np.mean(comp_areas)
                        
                        # Analyze stroke characteristics
                        horizontal_projection = np.sum(roi, axis=1)
                        vertical_projection = np.sum(roi, axis=0)
                        h_var_coef = np.std(horizontal_projection) / np.mean(horizontal_projection) if np.mean(horizontal_projection) > 0 else 0
                        v_var_coef = np.std(vertical_projection) / np.mean(vertical_projection) if np.mean(vertical_projection) > 0 else 0
                        
                        if (h_var_coef > 1.2 and v_var_coef > 1.0 and  # 4x stricter stroke variation
                            0.45 < comp_var < 0.95 and  # Tighter component variation
                            angle_variation > 0.8):  # Require more stroke direction variation
                            
                            density = cv2.countNonZero(roi) / (w * h)
                            if 0.16 < density < 0.17:  # Ultra strict density range
                                moments = cv2.moments(contour)
                                hu_moments = cv2.HuMoments(moments)
                                if hu_moments[0] < -2.8:  # Stricter shape characteristic
                                    # Extract signature region
                                    sign_region = original[y:y+h, x:x+w]
                                    
                                    # Convert to grayscale for finding content
                                    sign_gray = cv2.cvtColor(sign_region, cv2.COLOR_BGR2GRAY)
                                    _, sign_thresh = cv2.threshold(sign_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    
                                    # Find actual content boundaries
                                    coords = cv2.findNonZero(sign_thresh)
                                    if coords is not None:
                                        x_min, y_min, w_tight, h_tight = cv2.boundingRect(coords)
                                        sign_region = sign_region[y_min:y_min+h_tight, x_min:x_min+w_tight]
                                        
                                        # Save tightly cropped signature
                                        sign_filename = os.path.join(pred_dir, f"Sign{sign_count}.jpg")
                                        cv2.imwrite(sign_filename, sign_region)
                                        sign_count += 1
                                    
                                    cv2.rectangle(stamp_vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
    
    # Save visualization without displaying
    cv2.imwrite(os.path.join(pred_dir, "detection.jpg"), stamp_vis)

# Process all images in VIT_Dataset
dataset_path = "VIT_Dataset"
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(dataset_path, filename)
        process_image(image_path)
        print(f"Processed {filename}")
