import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "VIT_Dataset/kak-sostavit-dop-soglashenie-k-dogovoru-na-okazanie-uslug-obrazets.jpg"
image = cv2.imread(image_path)

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
for contour in stamp_contours:
    if cv2.contourArea(contour) > 500:  # Filter small noise
        x, y, w, h = cv2.boundingRect(contour)
        stamp_regions.append(image[y:y+h, x:x+w])

# Process signature contours
for contour in signature_contours:
    if cv2.contourArea(contour) > 500:  # Filter small noise
        x, y, w, h = cv2.boundingRect(contour)
        if not any(cv2.countNonZero(stamp_mask[y:y+h, x:x+w]) > 0.5 * w * h for stamp in stamp_regions):
            signature_regions.append(image[y:y+h, x:x+w])

# Save cropped regions
for i, stamp in enumerate(stamp_regions):
    cv2.imwrite(f'stamp_{i}.png', stamp)

for i, signature in enumerate(signature_regions):
    cv2.imwrite(f'signature_{i}.png', signature)

# Display results
fig, ax = plt.subplots(2, max(len(stamp_regions), len(signature_regions)), figsize=(15, 8))

for i, stamp in enumerate(stamp_regions):
    ax[0, i].imshow(cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB))
    ax[0, i].set_title(f"Stamp {i+1}")
    ax[0, i].axis("off")

for i, signature in enumerate(signature_regions):
    ax[1, i].imshow(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    ax[1, i].set_title(f"Signature {i+1}")
    ax[1, i].axis("off")

plt.tight_layout()
plt.show()