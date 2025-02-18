import pandas as pd
import cv2
import os

# Resimlerin bulunduğu klasörün tam yolu
image_folder = r"C:\Users\frknt\Desktop\Pose Detection\real_human_pose_dataset"  # Resimlerin olduğu klasör yolunu buraya yaz

# CSV dosyasını yükle
csv_file = "mpii_human_pose.csv"  # CSV dosyasının adı
data = pd.read_csv(csv_file)

# Keypoint'ler arasında bağlantılar (sıra önemli)
connections = [
    ("head top", "upper neck"),
    ("upper neck", "thorax"),
    ("thorax", "pelvis"),
    ("r hip", "r knee"),
    ("r knee", "r ankle"),
    ("l hip", "l knee"),
    ("l knee", "l ankle"),
    ("r shoulder", "r elbow"),
    ("r elbow", "r wrist"),
    ("l shoulder", "l elbow"),
    ("l elbow", "l wrist"),
    ("r shoulder", "thorax"),
    ("l shoulder", "thorax"),
    ("r hip", "pelvis"),
    ("l hip", "pelvis")
]

# Eksik keypoint'leri raporlamak için liste
missing_keypoints = []

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
        else:
            missing_keypoints.append((row['NAME'], name))  # Eksik keypoint kaydı

    # Keypoint'ler arasında bağlantı çizgilerini çizin
    for p1, p2 in connections:
        if pd.notna(row[f"{p1}_X"]) and pd.notna(row[f"{p1}_Y"]) and \
           pd.notna(row[f"{p2}_X"]) and pd.notna(row[f"{p2}_Y"]):
            x1, y1 = int(row[f"{p1}_X"]), int(row[f"{p1}_Y"])
            x2, y2 = int(row[f"{p2}_X"]), int(row[f"{p2}_Y"])
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Mavi çizgiler
        else:
            print(f"Bağlantı eksik: {p1} veya {p2} keypoint eksik.")

    # Görüntüyü göster
    cv2.imshow('Image with Keypoints and Connections', image)
    cv2.waitKey(0)

# Pencereyi kapat
cv2.destroyAllWindows()

# Eksik keypoint'leri raporla
if missing_keypoints:
    print("Eksik Keypoint'ler:")
    for image_name, keypoint in missing_keypoints:
        print(f"Resim: {image_name}, Eksik Keypoint: {keypoint}")
else:
    print("Hiçbir eksik keypoint bulunamadı.")
