import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class PoseDataset(Dataset):
    def __init__(self, images_path, keypoints_path, transform=None):
        self.images = np.load(images_path)
        self.keypoints = np.load(keypoints_path)
        self.transform = transform
        
        assert len(self.images) == len(self.keypoints), "Image and keypoint counts don't match!"
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # Get original image and keypoints
        image = self.images[idx].astype(np.float32)
        keypoints = self.keypoints[idx].astype(np.float32)
        
        # Store original dimensions for keypoint scaling
        orig_height, orig_width = image.shape[:2]
        
        if self.transform:
            # Convert to PIL Image for transforms
            image = transforms.ToPILImage()(torch.from_numpy(image.transpose(2, 0, 1)))
            image = self.transform(image)
            
            # Scale keypoints according to image transformations
            new_height, new_width = image.shape[1:3]
            keypoints[:, 0] *= (new_width / orig_width)
            keypoints[:, 1] *= (new_height / orig_height)
        else:
            # Just convert to tensor if no transforms
            image = torch.from_numpy(image.transpose(2, 0, 1)) / 255.0
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return image, keypoints
