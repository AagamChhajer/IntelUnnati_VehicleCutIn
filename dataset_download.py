import requests
import zipfile
import os

def download_dataset(url, destination_path):
    # Download the file from the URL
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # 1 KiB

    # Save the downloaded file to the specified path
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size):
            file.write(chunk)
    print(f"Downloaded {destination_path}")

def extract_zip_file(zip_path, extraction_path):
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
    print(f"Extracted {zip_path} to {extraction_path}")

# Define the URL for the dataset
dataset_url = "http://idd.insaan.iiit.ac.in/IDD_Temporal.zip"

# Define the path to save the downloaded file and the extraction path
download_path = "IDD_Temporal.zip"
extraction_path = "IDD_Temporal"

# Create the extraction directory if it doesn't exist
if not os.path.exists(extraction_path):
    os.makedirs(extraction_path)

# Download and extract the dataset
download_dataset(dataset_url, download_path)
extract_zip_file(download_path, extraction_path)
