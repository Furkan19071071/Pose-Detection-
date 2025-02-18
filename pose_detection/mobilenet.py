import torch
import torch.nn as nn
import torchvision.models as models

# MobileNet'i yükle ve özelleştir
class PoseMobileNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseMobileNet, self).__init__()
        # MobileNetV2'yi yükle
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # MobileNet'in son katmanını değiştir
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 512),  # Ara katman
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints * 2)  # 17 keypoint * (X, Y) = 34 çıkış
        )
    
    def forward(self, x):
        return self.mobilenet(x)

# Modeli oluştur
num_keypoints = 17
model = PoseMobileNet(num_keypoints=num_keypoints)

# Modeli GPU'ya taşıma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test için örnek girdi
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224).to(device)  # Örnek bir görüntü girdisi
    output = model(x)
    print("Çıktı boyutu:", output.shape)  # (1, 34)