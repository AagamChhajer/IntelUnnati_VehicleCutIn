import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load YOLOv5 model from torch.hub
yolov5_model = torch.hub.load('ultralytics/yolov7', 'yolov7s', pretrained=True)

# Function to preprocess images for YOLOv5
def prepare_image_for_inference(image):
    # Resize image to 640x640
    resized_image = cv2.resize(image, (640, 640))
    normalized_image = resized_image / 255.0  # Normalize to [0, 1]
    return normalized_image

# Function to perform object detection
def perform_object_detection(img_path):
    original_image = cv2.imread(img_path)  # Read image
    processed_image = prepare_image_for_inference(original_image)
    
    # Perform inference
    detection_results = yolov5_model(processed_image)
    
    # Print results
    detection_results.print()
    
    # Save results
    detection_results.save(save_dir='runs/detect/exp')  # Adjust save_dir as needed
    
    # Show results using OpenCV
    detected_image = detection_results.imgs[0]
    cv2.imshow('Detection Results', detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show results using Matplotlib
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test the model on a new image
test_image_path = 'path/to/your/test/image.jpg'
perform_object_detection(test_image_path)
