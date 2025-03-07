# Stamp Detection and Processing

This project implements a stamp detection and processing pipeline using YOLOv8 and OpenCV. It can detect stamps in documents and create background-less stamp images.

## Workflow

1. **Dataset Preparation**
   - Manual annotation of stamps in documents using Roboflow
   - Dataset extraction and preprocessing
   
2. **Model Training**
   - YOLOv8n model training for stamp detection
   - Training performed in `signdetect.ipynb`

3. **Inference Pipeline**
   - Document stamp detection using trained model
   - Bounding box extraction of detected stamps
   - Background removal using HSV color filtering for blue stamps


## HSV Color Ranges

The project uses specific HSV color ranges to isolate blue stamps:
- Hue: Blue color range
- Saturation: Color intensity
- Value: Brightness levels

## Model Details

- Base Model: YOLOv8n
- Task: Object Detection
- Classes: 1 (Stamp)
- Training Data: Custom annotated dataset

## Results

The pipeline successfully:
1. Detects stamps in documents
2. Extracts stamp regions
3. Creates clean, background-less stamp images

## Note

This project is specifically optimized for blue stamps in documents. Adjustments to HSV ranges may be needed for different colored stamps.
