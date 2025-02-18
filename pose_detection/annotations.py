import pandas as pd
import cv2
import os

# Resimlerin bulunduğu klasörün tam yolu
image_folder = r"C:\Users\frknt\Desktop\Pose Detection\real_human_pose_dataset"  # Resimlerin olduğu klasör yolunu buraya yaz

# CSV dosyasını yükle
csv_file = "mpii_human_pose.csv"  # CSV dosyasının adı
data = pd.read_csv(csv_file)

# Resimleri ve keypoint'leri eşleştirme
for index, row in data.iterrows():
    # Resim yolunu oluştur
    image_path = os.path.join(image_folder, row['NAME'])  # Resim klasörüyle dosya adını birleştir

    if not os.path.exists(image_path):
        print(f"Resim bulunamadı: {image_path}")
        continue

    # Resmi yükle
    image = cv2.imread(image_path)

    # Keypoint koordinatlarını çek
    keypoints = [
        ("r ankle", (row['r ankle_X'], row['r ankle_Y'])),
        ("r knee", (row['r knee_X'], row['r knee_Y'])),
        ("r hip", (row['r hip_X'], row['r hip_Y'])),
        ("l hip", (row['l hip_X'], row['l hip_Y'])),
        ("l knee", (row['l knee_X'], row['l knee_Y'])),
        ("l ankle", (row['l ankle_X'], row['l ankle_Y'])),
        ("pelvis", (row['pelvis_X'], row['pelvis_Y'])),
        ("thorax", (row['thorax_X'], row['thorax_Y'])),
        ("upper neck", (row['upper neck_X'], row['upper neck_Y'])),
        ("head top", (row['head top_X'], row['head top_Y'])),
        ("r wrist", (row['r wrist_X'], row['r wrist_Y'])),
        ("r elbow", (row['r elbow_X'], row['r elbow_Y'])),
        ("r shoulder", (row['r shoulder_X'], row['r shoulder_Y'])),
        ("l shoulder", (row['l shoulder_X'], row['l shoulder_Y'])),
        ("l elbow", (row['l elbow_X'], row['l elbow_Y'])),
        ("l wrist", (row['l wrist_X'], row['l wrist_Y'])),
    ]

    # Her bir keypoint'i resme çizin
    for name, (x, y) in keypoints:
        if pd.notna(x) and pd.notna(y):  # NaN değer kontrolü
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Yeşil noktalar

    # Görüntüyü göster
    cv2.imshow('Image with Keypoints', image)
    cv2.waitKey(0)

# Pencereyi kapat
cv2.destroyAllWindows()
