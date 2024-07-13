import torch
import cv2
from matplotlib import pyplot as plt

# Define the path to the trained model weights
model_weights_path = 'runs/train/exp/weights/best.pt'  # Adjust the path as necessary

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov7', 'custom', path=model_weights_path, source='local')

# Function to run inference on an image
def perform_inference(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Run inference
    detection_results = yolo_model(image)
    
    # Print results
    detection_results.print()
    
    # Save results
    detection_results.save(save_dir='runs/detect/exp')  # Adjust save_dir as needed
    
    # Show results using OpenCV
    result_image = detection_results.imgs[0]
    cv2.imshow('Detection Results', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show results using Matplotlib
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test the model on a new image
if __name__ == '__main__':
    test_img_path = 'path/to/your/test/image.jpg'  # Adjust the path as necessary
    perform_inference(test_img_path)
