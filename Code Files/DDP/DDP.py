import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import csv

import argparse

from tqdm.auto import tqdm

from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms

# Initialize distributed process group
def setup_ddp():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    # Set device for current process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def initialize_csv():
    csv_file = 'training_times.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["GPUs", "Batch Size", "Training Time (Seconds)"])

# Dataset path
DATASET_PATH = "/scratch/panchani.d/Hpc/dataset"

# Function to get image file paths
def get_image_paths():
    class_folders = [os.path.join(DATASET_PATH, d) for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    image_paths = []
    class_labels = []
    
    for class_folder in class_folders:
        label = os.path.basename(class_folder)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            if img_path.endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(img_path)
                class_labels.append(label)
    
    return image_paths, class_labels


class SkinDiseaseDataset(Dataset):
    def __init__(self, image_paths, class_labels, class_to_idx, transform=None, augment=False):
        self.image_paths = image_paths
        self.class_labels = class_labels
        self.class_to_idx = class_to_idx
        self.augment = augment
        
        # Define base transforms that all images must go through
        self.base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define augmentation transforms if needed
        self.augment_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Use provided transform if given, otherwise use base_transform
        self.transform = transform if transform is not None else self.base_transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            try:
                # Get image path and label
                img_path = self.image_paths[idx]
                class_name = self.class_labels[idx]
                
                # Convert string label to integer index
                label = self.class_to_idx[class_name]
                
                # Load and process image
                image = Image.open(img_path).convert('RGB')
                
                # Apply transformations
                if self.augment:
                    image = self.augment_transform(image)
                else:
                    image = self.transform(image)
                    
                return image, label
                
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"Error opening image: {img_path}. Trying another image.")
                attempts += 1
                if attempts >= max_attempts:
                    # Return a black image as fallback
                    print(f"Failed to load image after {max_attempts} attempts.")
                    dummy_img = torch.zeros(3, 224, 224)
                    return dummy_img, 0  # Return 0 as fallback label
                
                # Try a different random image
                idx = random.randint(0, len(self.image_paths) - 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size per GPU')
    return parser.parse_args()

def prepare_data(local_rank, batch_size):
    # Load dataset image paths
    image_paths, class_labels = get_image_paths()
    
    # Get unique class names
    unique_classes = sorted(set(class_labels))
    
    # Create mapping dictionary
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    # Split data
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths, class_labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = SkinDiseaseDataset(
        image_paths=train_image_paths, 
        class_labels=train_labels, 
        class_to_idx=class_to_idx, 
        # transform=transform,
        transform=None,
        augment=True
    )
    
    val_dataset = SkinDiseaseDataset(
        image_paths=val_image_paths, 
        class_labels=val_labels, 
        class_to_idx=class_to_idx, 
        # transform=transform,
        transform=None,
        augment=False
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if local_rank == 0:
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, train_sampler, val_sampler, len(unique_classes)

def train_model(local_rank, world_size, batch_size):

    if local_rank == 0:
        start_time = time()  # Start timing
    
    # Set device
    device = torch.device(f"cuda:{local_rank}")
    
    # Prepare data
    train_loader, val_loader, train_sampler, val_sampler, num_classes = prepare_data(local_rank, batch_size)
    
    # Load pre-trained DenseNet121
    # model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Remove the last layer and add a new classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=(local_rank == 0)
    )
    
    # Lists to store metrics (only on rank 0)
    if local_rank == 0:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
    
    # Training loop with validation
    num_epochs = 5
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if local_rank == 0:
            print(f'Starting epoch: {epoch + 1}')
            train_iter = tqdm(train_loader)
        else:
            train_iter = train_loader
        
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics (only for rank 0)
            if local_rank == 0:
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Gather statistics from all processes
        if local_rank == 0:
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Save training metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics (only for rank 0)
                if local_rank == 0:
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
        
        # Gather validation statistics
        if local_rank == 0:
            val_loss = val_running_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Save validation metrics
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.module.state_dict(), f'./DDP/checkpoint/best_skin_model_{world_size}_{batch_size}.pth')
                print(f"Model saved with validation accuracy: {val_acc:.2f}%")
            
            # Print statistics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 60)

    if local_rank == 0:
        training_time = time() - start_time  # Calculate training time
        print(f"Training completed in {training_time:.2f} seconds.")
        # Write to CSV
        with open('training_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([world_size, batch_size, training_time])
    
    # Final reporting and visualization (only on rank 0)
    if local_rank == 0:
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save metrics to file
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, num_epochs + 1)),
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        })
        metrics_df.to_csv('training_metrics.csv', index=False)
        
        # Plot training and validation metrics
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics_plot.png')
        
        # Load best model for final evaluation
        model.module.load_state_dict(torch.load(f'./DDP/checkpoint/best_densenet_skin_model_{world_size}_{batch_size}.pth'))
        
        # Function to compute accuracy
        def compute_accuracy(model, data_loader, device):
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            return accuracy
        
        final_val_acc = compute_accuracy(model, val_loader, device)
        print(f"Final validation accuracy: {final_val_acc:.2f}%")

def main():

    initialize_csv()
    args = parse_args()  # Parse command line arguments
    
    # Initialize distributed process group
    local_rank = setup_ddp()
    
    # Get world size
    world_size = dist.get_world_size()
    
    if local_rank == 0:
        print(f"Training with {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
    
    if local_rank == 0:
        print(f"Training with {world_size} GPUs")
    
    # Train the model
    # train_model(local_rank, world_size)
    train_model(local_rank, world_size, args.batch_size)
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()


# Terminal Command TO run the file
# torchrun --nproc_per_node=4 DDP.py
