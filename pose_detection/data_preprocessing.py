import os
import pandas as pd
import cv2
import numpy as np

# Ayarlar
IMAGE_SIZE = (256, 256)  # Resimleri yeniden boyutlandırma boyutları
IMAGE_FOLDER = r"C:\Users\frknt\Desktop\Pose Detection\real_human_pose_dataset"  # Resim klasörü yolu
CSV_FILE = "mpii_human_pose.csv"  # Keypoint verilerinin CSV dosyası
OUTPUT_DIR = "preprocessed_dataset"  # Ön işlenmiş verilerin kaydedileceği klasör

# Klasör oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_image(image_path, target_size):
    """Resmi verilen boyuta yeniden boyutlandırır."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Resim yüklenemedi: {image_path}")
        return None
    resized_image = cv2.resize(image, target_size)
    return resized_image

def normalize_keypoints(keypoints, original_width, original_height, target_width, target_height):
    """Keypoint koordinatlarını normalize eder."""
    normalized_keypoints = []
    for x, y in keypoints:
        if np.isnan(x) or np.isnan(y):  # Eğer NaN değerse (-1) ata
            normalized_keypoints.append((-1, -1))
        else:
            norm_x = (x / original_width) * target_width
            norm_y = (y / original_height) * target_height
            normalized_keypoints.append((norm_x, norm_y))
    return normalized_keypoints

def preprocess_data(image_folder, csv_file, output_dir, image_size):
    """Veri ön işleme işlemini gerçekleştirir."""
    data = pd.read_csv(csv_file)
    
    preprocessed_images = []
    preprocessed_keypoints = []

    for index, row in data.iterrows():
        image_name = row['NAME']
        image_path = os.path.join(image_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Resim bulunamadı: {image_path}")
            continue
        
        # Resmi yükle ve yeniden boyutlandır
        original_image = cv2.imread(image_path)
        original_height, original_width = original_image.shape[:2]
        resized_image = resize_image(image_path, image_size)
        if resized_image is None:
            continue
        
        # Keypoint verilerini çek ve normalize et
        keypoints = [
            (row['r ankle_X'], row['r ankle_Y']),
            (row['r knee_X'], row['r knee_Y']),
            (row['r hip_X'], row['r hip_Y']),
            (row['l hip_X'], row['l hip_Y']),
            (row['l knee_X'], row['l knee_Y']),
            (row['l ankle_X'], row['l ankle_Y']),
            (row['pelvis_X'], row['pelvis_Y']),
            (row['thorax_X'], row['thorax_Y']),
            (row['upper neck_X'], row['upper neck_Y']),
            (row['head top_X'], row['head top_Y']),
            (row['r wrist_X'], row['r wrist_Y']),
            (row['r elbow_X'], row['r elbow_Y']),
            (row['r shoulder_X'], row['r shoulder_Y']),
            (row['l shoulder_X'], row['l shoulder_Y']),
            (row['l elbow_X'], row['l elbow_Y']),
            (row['l wrist_X'], row['l wrist_Y']),
        ]
        
        normalized_keypoints = normalize_keypoints(
            keypoints, original_width, original_height, image_size[0], image_size[1]
        )
        
        # Ön işlenmiş verileri listelere ekle
        preprocessed_images.append(resized_image)
        preprocessed_keypoints.append(normalized_keypoints)
    
    # Ön işlenmiş verileri kaydet
    np.save(os.path.join(output_dir, "images.npy"), np.array(preprocessed_images))
    np.save(os.path.join(output_dir, "keypoints.npy"), np.array(preprocessed_keypoints))
    print("Veri ön işleme tamamlandı. Veriler kaydedildi.")

# Fonksiyonu çalıştır
preprocess_data(IMAGE_FOLDER, CSV_FILE, OUTPUT_DIR, IMAGE_SIZE)
