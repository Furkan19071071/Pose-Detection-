import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_loader import PoseDataset  # Veri seti tanımı
from keypoint_mobilenet import KeypointMobileNet  # Model tanımı
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar için

# Debugging function to print tensor shapes
# İlk 10 görüntüyü çizdirme
# def plot_first_10_images(train_loader):
#     for i, (images, keypoints) in enumerate(train_loader):
#         if i >= 10:  # İlk 10 batch
#             break
#         for j in range(images.size(0)):
#             plot_keypoints(images[j].cpu().numpy(), keypoints[j].cpu().numpy())

# def plot_keypoints(image, keypoints):
#     """
#     Görüntü üzerinde keypoint'leri çizdiren fonksiyon.
#     :param image: Görüntü (NumPy array, C x H x W)
#     :param keypoints: Keypointler (NumPy array, N x 2)
#     """
#     image = image.transpose(1, 2, 0)  # Kanal boyutunu sonrasına alıyoruz
#     plt.imshow(image)  # Görüntüyü ekrana bastır
#     for keypoint in keypoints:
#         x, y = keypoint
#         plt.scatter(x, y, color='red', marker='x')  # Keypoint'leri kırmızı ile işaretle
#     plt.show()

# Paths
images_path = "preprocessed_dataset/images.npy"
keypoints_path = "preprocessed_dataset/keypoints.npy"

# Transform işlemleri
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Veri setini yükle
full_dataset = PoseDataset(images_path, keypoints_path, transform)

# Eğitim ve validasyon setlerini böl
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
#plot_first_10_images(train_loader)

# Model
num_keypoints = 16  # Keypoint sayısı
model = KeypointMobileNet(num_keypoints=num_keypoints).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function ve optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Accuracy calculation
def calculate_accuracy(outputs, keypoints, threshold=5.0):
    outputs = outputs.view(-1, num_keypoints, 2)
    keypoints = keypoints.view(-1, num_keypoints, 2)
    
    distances = torch.norm(outputs - keypoints, dim=2)
    correct_predictions = distances < threshold
    accuracy = correct_predictions.float().mean().item()
    return accuracy

# Eğitim döngüsü
num_epochs = 50
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    print(f"Epoch [{epoch+1}/{num_epochs}]\n")
    
    # Progress bar for training
    train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    
    for batch_idx, (images, keypoints) in train_bar:
        images = images.float().to(device)
        keypoints = keypoints.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.view(-1, num_keypoints, 2)
        keypoints = keypoints.view(-1, num_keypoints, 2)
        
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    # Validasyon
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    
    val_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (images, keypoints) in val_bar:
            images = images.float().to(device)
            keypoints = keypoints.float().to(device)
            
            outputs = model(images)
            outputs = outputs.view(-1, num_keypoints, 2)
            keypoints = keypoints.view(-1, num_keypoints, 2)
            
            loss = criterion(outputs, keypoints)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, keypoints)
            val_bar.set_postfix(loss=loss.item())

    val_accuracy /= len(val_loader)
    best_val_accuracy = max(best_val_accuracy, val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}]\n"
          f"Train Loss: {train_loss / len(train_loader):.4f}\n"
          f"Val Loss: {val_loss / len(val_loader):.4f}\n"
          f"Val Accuracy: {val_accuracy * 100:.2f}%")

# Modeli kaydet
torch.save(model.state_dict(), "pose_detection_model.pth")
print(f"En İyi Validation Accuracy: {best_val_accuracy * 100:.2f}%")
print("Model kaydedildi.")
