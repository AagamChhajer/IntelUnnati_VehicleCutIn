import torch
from pathlib import Path

# Set the paths
config_file = 'idd_temporal.yaml'  # Path to the dataset configuration file
best_weights_path = 'runs/train/exp/weights/best.pt'  # Path to the best weights after training

# Define the device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv5 model
custom_model = torch.hub.load('ultralytics/yolov7', 'custom', path=best_weights_path, source='local')

# Function to evaluate the model
def evaluate_model(custom_model, config_file):
    # Run validation
    eval_results = custom_model.val(data=config_file, batch_size=16, imgsz=640, device=device)
    
    # Extract and print results
    eval_metrics = {
        'Precision': eval_results['metrics/precision'],
        'Recall': eval_results['metrics/recall'],
        'mAP_0.5': eval_results['metrics/mAP_0.5'],
        'mAP_0.5:0.95': eval_results['metrics/mAP_0.5:0.95']
    }
    
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return eval_metrics

# Run the evaluation
if __name__ == '__main__':
    metrics = evaluate_model(custom_model, config_file)
    print("Evaluation completed. Metrics:", metrics)
