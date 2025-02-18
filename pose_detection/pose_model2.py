import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_loader import PoseDataset
from keypoint_mobilenet2 import KeypointMobileNet
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """Metrics görselleştirme fonksiyonu"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Accuracy over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Paths
images_path = "preprocessed_dataset/images.npy"
keypoints_path = "preprocessed_dataset/keypoints.npy"

# Transform işlemleri - Augmentation eklendi
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Veri setini yükle
full_dataset = PoseDataset(images_path, keypoints_path, transform=None)  # Transform None olarak başlatılıyor

# Eğitim ve validasyon setlerini böl
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Transform'ları ayrı ayrı uygula
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Model
num_keypoints = 16
model = KeypointMobileNet(num_keypoints=num_keypoints).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function ve optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

def calculate_accuracy(outputs, keypoints, threshold=5.0):
    outputs = outputs.view(-1, num_keypoints, 2)
    keypoints = keypoints.view(-1, num_keypoints, 2)
    
    distances = torch.norm(outputs - keypoints, dim=2)
    correct_predictions = distances < threshold
    accuracy = correct_predictions.float().mean().item()
    return accuracy

# Eğitim için gerekli değişkenler
num_epochs = 20
best_val_accuracy = 0.0
patience = 5
no_improve_epochs = 0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    
    train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (images, keypoints) in train_bar:
        images = images.float().to(device)
        keypoints = keypoints.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.view(-1, num_keypoints, 2)
        keypoints = keypoints.view(-1, num_keypoints, 2)
        
        loss = criterion(outputs, keypoints)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy += calculate_accuracy(outputs, keypoints)
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    avg_train_accuracy = train_accuracy / len(train_loader)
    
    # Validation
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

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)
    
    # Metrics kaydetme
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(avg_train_accuracy)
    val_accuracies.append(avg_val_accuracy)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # En iyi modeli kaydet
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_accuracy': avg_val_accuracy,
        }, "best_pose_detection_model.pth")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy*100:.2f}%")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy*100:.2f}%")
    
    # Early stopping
    if no_improve_epochs >= patience:
        print("\nEarly stopping triggered!")
        break

# Metrikleri görselleştir
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

print(f"\nEn İyi Validation Accuracy: {best_val_accuracy*100:.2f}%")
print("Eğitim tamamlandı ve metrikler kaydedildi.")