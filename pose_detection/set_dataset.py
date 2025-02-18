import pandas as pd
import cv2
import os
import shutil

# Resimlerin bulunduğu klasörün tam yolu
image_folder = r"C:\Users\frknt\Desktop\Pose Detection\mpii_pose_dataset\mpii_images\images"

# Yeni bir klasör oluştur (eşleşen resimler için)
output_folder = r"C:\Users\frknt\Desktop\Pose Detection\real_human_pose_dataset"
os.makedirs(output_folder, exist_ok=True)

# CSV dosyasını yükle
csv_file = "mpii_human_pose.csv"
data = pd.read_csv(csv_file)

# Eşleşen resim sayacı
matched_count = 0

# Resimleri ve keypoint'leri eşleştirme
for index, row in data.iterrows():
    # Resim yolunu oluştur
    image_path = os.path.join(image_folder, row['NAME'])

    if os.path.exists(image_path):
        # Eşleşen resmi hedef klasöre kopyala
        shutil.copy(image_path, os.path.join(output_folder, row['NAME']))
        matched_count += 1
    else:
        print(f"Eşleşmeyen resim: {row['NAME']}")

# İşlem tamamlandıktan sonra eşleşen resim sayısını yazdır
print(f"Eşleşen resim sayısı: {matched_count}")
print(f"Eşleşen resimler {output_folder} klasörüne kopyalandı.")
