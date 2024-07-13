import os
import cv2
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "IDD_Temporal"
images_dir = os.path.join(dataset_dir, "Images")
annotations_dir = os.path.join(dataset_dir, "Annotations")

# Function to load and preprocess an image
def preprocess_image(image_path, resize_dim=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(resize_dim)
    img = np.array(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to load annotations
def read_annotations(annotation_file_path):
    with open(annotation_file_path, 'r') as file:
        annotation_data = json.load(file)
    return annotation_data

# Prepare dataset lists
def gather_files(directory, extensions):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                file_list.append(os.path.join(root, file))
    return file_list

image_files = gather_files(images_dir, (".jpg", ".png"))
annotation_files = gather_files(annotations_dir, (".json",))

# Ensure images and annotations match
image_files.sort()
annotation_files.sort()

# Load and preprocess data
def load_data(image_paths, annotation_paths):
    imgs, labels = [], []
    for img_path, ann_path in zip(image_paths, annotation_paths):
        img = preprocess_image(img_path)
        annotation = read_annotations(ann_path)
        bboxes = annotation.get('bboxes', [])
        imgs.append(img)
        labels.append(bboxes)
    return np.array(imgs), np.array(labels)

images, labels = load_data(image_files, annotation_files)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save preprocessed data if needed
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
