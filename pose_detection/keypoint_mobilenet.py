import torch
import torch.nn as nn
from torchvision import models

class KeypointMobileNet(nn.Module):
    def __init__(self, num_keypoints, pretrained=True):
        super(KeypointMobileNet, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        
        # Feature extractor
        self.features = mobilenet.features
        
        # Daha karmaşık regressor
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints * 2)
        )
        
        # Ağırlık başlatma
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.view(-1, self.num_keypoints, 2)

if __name__ == "__main__":
    model = KeypointMobileNet(num_keypoints=16)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (1, 16, 2)
