"""
Train product embedding model using triplet loss
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import random
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import config

class ProductDataset(Dataset):
    def __init__(self, reference_dir: str, transform=None):
        self.reference_dir = reference_dir
        self.transform = transform
        self.classes = []
        self.image_paths = []
        self.labels = []
        
        # Load dataset
        for class_name in os.listdir(reference_dir):
            class_dir = os.path.join(reference_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_id = len(self.classes)
            self.classes.append(class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_id)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get anchor image
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        
        # Get positive image (same class)
        positive_paths = [
            p for i, p in enumerate(self.image_paths)
            if self.labels[i] == anchor_label and p != anchor_path
        ]
        positive_path = random.choice(positive_paths) if positive_paths else anchor_path
        
        # Get negative image (different class)
        negative_paths = [
            p for i, p in enumerate(self.image_paths)
            if self.labels[i] != anchor_label
        ]
        negative_path = random.choice(negative_paths)
        
        # Load and transform images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative, anchor_label

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def train_embedding_model(
    reference_dir: str,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    margin: float = 1.0,
    device: str = None,
    progress_callback = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = ProductDataset(reference_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Identity()  # Remove classification layer
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (anchors, positives, negatives, _) in enumerate(progress_bar):
            # Move data to device
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)
            
            # Calculate loss
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            batch_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': batch_loss})
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, batch_loss)
        
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}')
    
    # Save the model
    model_path = os.path.join(os.path.dirname(reference_dir), 'product_embeddings.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes
    }, model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    train_embedding_model(config.REFERENCE_DIR) 