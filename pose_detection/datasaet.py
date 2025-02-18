import os
import urllib.request
import tarfile
import json
from scipy.io import loadmat

def download_mpii_pose_dataset(save_path='./mpii_pose_dataset'):
    # Create necessary directories
    os.makedirs(save_path, exist_ok=True)

    # MPII Pose Dataset URLs
    images_url = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
    annotations_url = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_1.mat'

    # Download paths
    images_path = os.path.join(save_path, 'mpii_images.tar.gz')
    annotations_path = os.path.join(save_path, 'mpii_annotations.mat')

    try:
        # Download images
        print("Downloading MPII images...")
        urllib.request.urlretrieve(images_url, images_path)

        # Download annotations
        print("Downloading MPII annotations...")
        urllib.request.urlretrieve(annotations_url, annotations_path)

        # Extract image files
        print("Extracting images...")
        with tarfile.open(images_path, 'r:gz') as tar:
            tar.extractall(path=save_path)

        # Remove compressed images file
        os.remove(images_path)

        print("Converting annotations to JSON format...")
        # Convert .mat to JSON
        mat_file = annotations_path
        json_file = os.path.join(save_path, 'mpii_human_pose_v1_u12_1.json')
        convert_mat_to_json(mat_file, json_file)

        # Remove .mat file after conversion
        os.remove(mat_file)

        print("Download, extraction, and conversion completed successfully!")

        # Provide a summary of the dataset
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            print(f"Annotation keys: {list(annotations.keys())}")
        else:
            print("JSON annotation file not found. Please check for errors.")

    except Exception as e:
        print(f"An error occurred: {e}")

def convert_mat_to_json(mat_file, json_file):
    try:
        data = loadmat(mat_file)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Converted {mat_file} to {json_file} successfully.")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Download the dataset
download_mpii_pose_dataset()
