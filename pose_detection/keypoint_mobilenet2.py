import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union, List
import numpy as np

class KeypointMobileNet(nn.Module):
    def __init__(
        self, 
        num_keypoints: int,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
        trainable_layers: int = 6
    ):
        """
        Initialize KeypointMobileNet model.
        
        Args:
            num_keypoints (int): Number of keypoints to detect
            pretrained (bool): Whether to use pretrained weights for backbone
            dropout_rate (float): Dropout rate for regularization
            freeze_backbone (bool): Whether to freeze backbone layers
            trainable_layers (int): Number of last layers to train in backbone
        """
        super(KeypointMobileNet, self).__init__()
        
        # Store parameters
        self.num_keypoints = num_keypoints
        self.dropout_rate = dropout_rate
        
        # Load MobileNetV2 with pretrained weights
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        
        # Feature extractor
        self.features = mobilenet.features
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in list(self.features.parameters())[:-trainable_layers]:
                param.requires_grad = False
        
        # Regressor with dropout and batch normalization
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, num_keypoints * 2)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model info
        self._print_model_info()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization."""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                # Kaiming/He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """Print model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print("\nModel Architecture Information:")
        print(f"Number of keypoints: {self.num_keypoints}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Dropout rate: {self.dropout_rate}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_keypoints, 2)
        """
        # Feature extraction
        x = self.features(x)
        
        # Regress keypoints
        x = self.regressor(x)
        
        # Reshape to (batch_size, num_keypoints, 2)
        x = x.view(-1, self.num_keypoints, 2)
        
        return x
    
    def predict_keypoints(
        self, 
        image: torch.Tensor, 
        return_confidence: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Predict keypoints for a single image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (3, height, width)
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            np.ndarray: Predicted keypoints of shape (num_keypoints, 2)
            Optional[np.ndarray]: Confidence scores if return_confidence is True
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # Move to same device as model
            image = image.to(next(self.parameters()).device)
            
            # Forward pass
            output = self(image)
            
            # Convert to numpy
            keypoints = output[0].cpu().numpy()
            
            if return_confidence:
                # Calculate confidence based on feature activations
                feature_maps = self.features(image)
                confidence = torch.mean(feature_maps, dim=(2, 3))
                confidence = torch.sigmoid(confidence)[0].cpu().numpy()
                return keypoints, confidence
            
            return keypoints
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

if __name__ == "__main__":
    # Model test
    num_keypoints = 16
    model = KeypointMobileNet(
        num_keypoints=num_keypoints,
        pretrained=True,
        dropout_rate=0.2,
        freeze_backbone=True,
        trainable_layers=6
    )
    
    # Test with random input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Print shapes
    print("\nModel Test Results:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test predict_keypoints method
    single_image = torch.randn(3, 224, 224)
    keypoints = model.predict_keypoints(single_image)
    keypoints_with_conf = model.predict_keypoints(single_image, return_confidence=True)
    
    print("\nPredict Keypoints Test:")
    print(f"Single image keypoints shape: {keypoints.shape}")
    if isinstance(keypoints_with_conf, tuple):
        print(f"Keypoints with confidence shapes: {keypoints_with_conf[0].shape}, {keypoints_with_conf[1].shape}")
    
    # Test if output shapes are correct
    expected_shape = (batch_size, num_keypoints, 2)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
    assert keypoints.shape == (num_keypoints, 2), "Single image prediction shape is incorrect"
    
    print("\nAll tests passed successfully!")