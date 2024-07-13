import os
import torch
from pathlib import Path

# Set the paths
config_path = 'idd_temporal.yaml'  # Path to the dataset configuration file
initial_weights = 'yolov7s.pt'  # Pretrained weights to start with
num_epochs = 50  # Number of epochs
train_batch_size = 16  # Batch size

# Train YOLOv5 model
def train_yolov5_model(config_path, initial_weights, num_epochs, train_batch_size):
    # Load the YOLOv5 model from torch hub
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Train the model
    yolo_model.train()
    yolo_model = yolo_model.to(device)  # Move model to GPU if available

    # Setup training
    train_results = yolo_model.train(data=config_path, epochs=num_epochs, batch_size=train_batch_size, weights=initial_weights)

    return train_results

# Define device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run the training
if __name__ == '__main__':
    results = train_yolov5_model(config_path, initial_weights, num_epochs, train_batch_size)
    print("Training completed. Results:", results)
