import streamlit as st
import cv2
import numpy as np
from PIL import Image
import zipfile
import os
from io import BytesIO
from ultralytics import YOLO

# Configure page
st.set_page_config(
    page_title="Stamp Processor",
    page_icon="🔍",
    layout="wide"
)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Add a threshold slider in the sidebar
st.sidebar.header("HSV Threshold Control")
threshold = st.sidebar.slider("Blue Detection Threshold", 0, 100, 50, 
                            help="Adjust to fine-tune the blue detection")

# Map the threshold to HSV ranges
def get_hsv_thresholds(threshold_value):
    # As threshold increases:
    # - Higher threshold = more specific/selective blue detection
    # - Lower threshold = wider range of blues detected
    
    h_base = 110  # Center of blue hue
    h_range = int(40 * (100 - threshold_value) / 100) + 5  # Ranges from 5 to 45
    h_min = max(90, h_base - h_range)
    h_max = min(130, h_base + h_range)
    
    s_min = max(10, int(threshold_value * 0.5))  # Minimum saturation increases with threshold
    v_min = max(10, int(threshold_value * 0.5))  # Minimum value increases with threshold
    
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, 255, 255])
    
    return lower_hsv, upper_hsv

# Get current HSV thresholds based on slider
lower_blue, upper_blue = get_hsv_thresholds(threshold)

# Store uploaded files in session state to avoid re-uploading
if 'uploaded_files_cache' not in st.session_state:
    st.session_state.uploaded_files_cache = []
    st.session_state.original_images = []
    st.session_state.detection_results = []

def process_image(img_data, apply_mask=True):
    """Process image and return original and processed versions"""
    nparr = np.frombuffer(img_data.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = img.copy()  # Keep a copy of the original full image
    
    results = model(img)
    
    processed_stamps = []
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cropped_img = img[y1:y2, x1:x2]
            
            if apply_mask:
                # Create mask and transparency using current threshold values
                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = np.where(mask == 255, 255, 0)  # Apply transparency
                
                processed_stamps.append({
                    'region': [x1, y1, x2, y2],  # Store the coordinates
                    'processed': bgra
                })
            
    return original_img, processed_stamps, results

# Main app
st.title("Stamp Processor")
st.markdown("Upload images to detect stamps from documents")

upload_option = st.radio("Select input type:", 
                        ["Single Image", "Multiple Images", "Zip File"])

uploaded_files = []
new_upload = False

if upload_option == "Single Image":
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if file:
        if not st.session_state.uploaded_files_cache or file.name != st.session_state.uploaded_files_cache[0].get('name'):
            uploaded_files = [file]
            new_upload = True
elif upload_option == "Multiple Images":
    files = st.file_uploader("Upload images (max 10)", 
                                    type=["jpg", "png", "jpeg"], 
                                    accept_multiple_files=True)[:10]
    if files and len(files) > 0:
        uploaded_files = files
        new_upload = True
elif upload_option == "Zip File":
    zip_file = st.file_uploader("Upload zip file", type="zip")
    if zip_file:
        # Check if this is a new upload
        if not st.session_state.uploaded_files_cache or (
            len(st.session_state.uploaded_files_cache) > 0 and 
            zip_file.name != st.session_state.uploaded_files_cache[0].get('zip_name')):
            new_upload = True
            with zipfile.ZipFile(zip_file, 'r') as zf:
                for i, name in enumerate(zf.namelist()):
                    if i >= 10:  # Limit to 10 images
                        break
                    if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with zf.open(name) as file:
                            uploaded_files.append({
                                "name": name, 
                                "data": file.read(),
                                "zip_name": zip_file.name
                            })

# Update the cache if we have new uploads
if new_upload:
    st.session_state.uploaded_files_cache = []
    st.session_state.original_images = []
    st.session_state.detection_results = []
    
    for file in uploaded_files:
        if isinstance(file, dict):  # From zip
            st.session_state.uploaded_files_cache.append(file)
        else:
            file_data = file.read()
            st.session_state.uploaded_files_cache.append({
                "name": file.name,
                "data": file_data
            })
    
    # Process the new uploads to get detections
    for file_data in st.session_state.uploaded_files_cache:
        try:
            data_bytes = BytesIO(file_data["data"])
            original_img, _, results = process_image(img_data=data_bytes, apply_mask=False)
            
            # Store the original image and detection results
            st.session_state.original_images.append(original_img)
            st.session_state.detection_results.append(results)
            
        except Exception as e:
            st.error(f"Error processing {file_data['name']}: {str(e)}")

# If we have data in the session state, process it with the current threshold
if st.session_state.uploaded_files_cache and st.session_state.original_images:
    processed_stamps = []
    
    # Display Original Images
    st.write("## Original Images")
    cols = st.columns(min(3, len(st.session_state.original_images)))
    
    for idx, orig_img in enumerate(st.session_state.original_images):
        col_idx = idx % len(cols)
        with cols[col_idx]:
            rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption=f"Original Image {idx+1}", use_container_width=True)
    
    # Process the stamps using current threshold
    for idx, (orig_img, results) in enumerate(zip(st.session_state.original_images, st.session_state.detection_results)):
        img_stamps = []
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cropped_img = orig_img[y1:y2, x1:x2]
                
                # Apply current threshold to get mask
                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = np.where(mask == 255, 255, 0)  # Apply transparency
                
                img_stamps.append(bgra)
        
        processed_stamps.extend(img_stamps)
    
    # Display Processed Stamps
    if processed_stamps:
        st.write("## Processed Stamps")
        cols = st.columns(min(3, len(processed_stamps)))
        
        for idx, stamp in enumerate(processed_stamps):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                rgba_img = cv2.cvtColor(stamp, cv2.COLOR_BGRA2RGBA)
                st.image(rgba_img, caption=f"Processed Stamp {idx+1}", use_container_width=True)
        
        # Download functionality
        st.write("### Download Results")
        download_images = []
        
        for idx, stamp in enumerate(processed_stamps):
            buffered = BytesIO()
            rgba_img = cv2.cvtColor(stamp, cv2.COLOR_BGRA2RGBA)
            Image.fromarray(rgba_img).save(buffered, format="PNG")
            download_images.append((f"stamp_{idx+1}_processed.png", buffered.getvalue()))
        
        if len(download_images) == 1:
            st.download_button(
                label="Download Processed Stamp",
                data=download_images[0][1],
                file_name=download_images[0][0],
                mime="image/png"
            )
        else:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                for name, data in download_images:
                    zf.writestr(name, data)
            
            st.download_button(
                label="Download All Stamps as ZIP",
                data=zip_buffer.getvalue(),
                file_name="processed_stamps.zip",
                mime="application/zip"
            )
    else:
        st.warning("No stamps detected in uploaded images")
else:
    st.info("Please upload images to begin processing")
