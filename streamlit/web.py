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

# Define HSV range for blue color
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

def process_image(img_data):
    """Process image and return original and processed versions"""
    nparr = np.frombuffer(img_data.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    
    processed_images = []
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cropped_img = img[y1:y2, x1:x2]
            
            # Create mask and transparency
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = np.where(mask == 255, 255, 0)
            
            processed_images.append({
                'original': cropped_img,
                'processed': bgra
            })
    
    return processed_images

# Main app
st.title("YOLO Stamp Processor with Transparency")
st.markdown("Upload images to detect stamps and make blue parts transparent")

upload_option = st.radio("Select input type:", 
                        ["Single Image", "Multiple Images", "Zip File"])

uploaded_files = []
if upload_option == "Single Image":
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if file:
        uploaded_files = [file]
elif upload_option == "Multiple Images":
    uploaded_files = st.file_uploader("Upload images (max 10)", 
                                    type=["jpg", "png", "jpeg"], 
                                    accept_multiple_files=True)[:10]
elif upload_option == "Zip File":
    zip_file = st.file_uploader("Upload zip file", type="zip")
    if zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for i, name in enumerate(zf.namelist()):
                if i >= 10:  # Limit to 10 images
                    break
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with zf.open(name) as file:
                        uploaded_files.append({"name": name, "data": file.read()})

if uploaded_files:
    all_processed = []
    original_images = []
    processed_images = []
    
    with st.spinner("Processing images..."):
        for file in uploaded_files:
            try:
                if isinstance(file, dict):  # From zip
                    file_name = file["name"]
                    file_data = BytesIO(file["data"])
                else:  # From regular upload
                    file_name = file.name
                    file.seek(0)
                    file_data = file
                
                processed_results = process_image(img_data=file_data)
                all_processed.extend(processed_results)
                
                # Save original image for display
                original_img_bytes = BytesIO()
                Image.open(file_data).save(original_img_bytes, format="JPEG")
                original_images.append(original_img_bytes.getvalue())
                
                # Save processed images for display
                for pair in processed_results:
                    processed_rgb = cv2.cvtColor(pair['processed'], cv2.COLOR_BGRA2RGBA)
                    buffered_processed = BytesIO()
                    Image.fromarray(processed_rgb).save(buffered_processed, format="PNG")
                    processed_images.append(buffered_processed.getvalue())
                
            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
    
    if all_processed:
        # Display Original Images Carousel (Smaller Size)
        st.write("## Original Images")
        for idx, img_bytes in enumerate(original_images):
            st.image(img_bytes, caption=f"Original Image {idx+1}", use_container_width=False, width=300)  # Set width to 300 pixels
        
        # Display Processed Images Carousel (Smaller Size)
        st.write("## Processed Stamps")
        for idx, img_bytes in enumerate(processed_images):
            st.image(img_bytes, caption=f"Processed Stamp {idx+1}", use_container_width=False, width=300)  # Set width to 300 pixels
        
        # Download functionality
        st.write("### Download Results")
        download_images = []
        
        for idx, img_bytes in enumerate(processed_images):
            download_images.append((f"stamp_{idx+1}_processed.png", img_bytes))
        
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
