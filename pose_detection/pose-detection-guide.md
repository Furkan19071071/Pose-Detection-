# Pose Detection Projesi
#210202067 Furksn TÜRKOĞLU
## Proje Hakkında
Bu proje, MobileNetV2 kullanarak insan vücudundaki 16 anahtar noktayı tespit eden bir yapay zeka sistemidir. Yüksek performans ve doğruluk hedeflenerek geliştirilmiştir.

## Kurulum
```bash
# Gerekli kütüphaneler
pip install torch torchvision numpy matplotlib tqdm
```

## Proje Yapısı
```
pose-detection/
├── preprocessed_dataset/
│   ├── images.npy      # Eğitim görüntüleri
│   └── keypoints.npy   # Keypoint koordinatları
├── data_loader.py      # Veri yükleme ve işleme
├── keypoint_mobilenet.py # Model mimarisi
└── pose_model.py       # Eğitim kodu
```

## Kullanım

### 1. Veri Hazırlama
- Görüntülerinizi ve keypoint verilerinizi numpy array formatında kaydedin
- `preprocessed_dataset` klasörüne yerleştirin
- Veriler otomatik olarak %80 eğitim, %20 test olarak bölünür

### 2. Model Eğitimi
```python
python pose_model.py
```

### 3. Görselleştirme
Eğitim sırasında sonuçları görmek için:
```python
# pose_model.py içindeki yorum satırını kaldırın
plot_first_10_images(train_loader)
```

## Teknik Detaylar

### Model Parametreleri
- Batch size: 16
- Learning rate: 0.001
- Epoch sayısı: 20
- Keypoint sayısı: 16
- Doğruluk eşiği: 5.0 piksel

### Performans Ölçümü
- Loss function: MSE (Mean Squared Error)
- Accuracy: 5 piksel içindeki tahmin oranı
- Her epoch sonunda validation metrikleri gösterilir

### Model Çıktısı
- Eğitim tamamlandığında model `pose_detection_model.pth` olarak kaydedilir
- Her keypoint için (x,y) koordinatları üretilir

## İpuçları
- GPU kullanımı önerilir
- Eğitim parametrelerini verilerinize göre ayarlayabilirsiniz
- Görselleştirme ile tahminlerin doğruluğunu kontrol edin

## Örnek Kullanım
```python
# Model yükleme
model = KeypointMobileNet(num_keypoints=16)
model.load_state_dict(torch.load('pose_detection_model.pth'))

# Tahmin
with torch.no_grad():
    predictions = model(image)
```
