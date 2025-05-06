import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.models as models
import time
import argparse
from tqdm import tqdm
import json
from pathlib import Path

# Import your dataset classes and utilities
from DDP import SkinDiseaseDataset, get_image_paths
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='EfficientNet-B3 Optimization with Multi-GPU Support')
    parser.add_argument('--dataset_path', type=str, default="/scratch/panchani.d/Hpc/dataset", 
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training (default: 64, per GPU)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, default="efficientnet_results", 
                        help='Directory to save results')
    parser.add_argument('--num_gpus', type=int, default=4, 
                        help='Number of GPUs to use (default: 4)')
    return parser.parse_args()

def prepare_data(batch_size, num_gpus=1):
    # Load dataset image paths
    image_paths, class_labels = get_image_paths()  # No arguments to fix the error
    
    # Get unique class names
    unique_classes = sorted(set(class_labels))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    # Split dataset
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths, class_labels, test_size=0.2, random_state=42, stratify=class_labels
    )
    
    # Create transforms
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SkinDiseaseDataset(
        image_paths=train_image_paths, 
        class_labels=train_labels,
        class_to_idx=class_to_idx,
        transform=train_transform
    )
    
    val_dataset = SkinDiseaseDataset(
        image_paths=val_image_paths, 
        class_labels=val_labels,
        class_to_idx=class_to_idx,
        transform=val_transform
    )
    
    # Adjust batch size for DataParallel (the effective batch size will be batch_size * num_gpus)
    effective_batch_size = batch_size
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(unique_classes)

def load_efficientnet_b3(num_classes):
    """Load EfficientNet-B3 model."""
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, mixed_precision=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use the updated API for GradScaler to fix the deprecation warning
    scaler = torch.amp.GradScaler() if mixed_precision else None
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if mixed_precision:
            # Use the updated API for autocast to fix the deprecation warning
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device, mixed_precision=False):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if mixed_precision:
                # Use the updated API for autocast to fix the deprecation warning
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def train_and_evaluate(learning_rate, mixed_precision, args):
    """Train and evaluate a model with the given learning rate and mixed precision setting."""
    # Clear CUDA cache before starting to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        available_gpus = min(args.num_gpus, torch.cuda.device_count())
        print(f"Using {available_gpus} GPUs with batch size {args.batch_size} per GPU")
        print(f"Total effective batch size: {args.batch_size * available_gpus}")
    else:
        device = torch.device("cpu")
        available_gpus = 0
        print("CUDA not available, using CPU")
    
    # Enable cudNN benchmark
    torch.backends.cudnn.benchmark = True
    
    # Prepare data
    train_loader, val_loader, num_classes = prepare_data(args.batch_size, available_gpus)
    
    # Load model
    model = load_efficientnet_b3(num_classes)
    
    # Use DataParallel if multiple GPUs are available
    if available_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(available_gpus)))
    
    model = model.to(device)
    
    # Print model size to help with debugging memory issues
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model has {model_size:,} parameters")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer - always using adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # For timing
    start_time = time.time()
    epoch_times = []
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, mixed_precision
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, mixed_precision)
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Measure epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 60)
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Create a unique identifier for this configuration
    config_id = f"lr{learning_rate}_{'mp' if mixed_precision else 'fp32'}"
    
    # Prepare results dictionary
    results = {
        'learning_rate': learning_rate,
        'mixed_precision': mixed_precision,
        'metrics': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'epoch_times': epoch_times,
            'total_time': total_time,
            'final_train_loss': train_losses[-1],
            'final_train_acc': train_accuracies[-1],
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accuracies[-1],
            'best_val_acc': max(val_accuracies),
            'avg_epoch_time': sum(epoch_times) / len(epoch_times)
        }
    }
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, args.epochs + 1)),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'epoch_time': epoch_times
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    
    # Save metrics to CSV
    metrics_df.to_csv(f"{args.output_dir}/metrics/{config_id}.csv", index=False)
    
    # Save model checkpoint for best validation accuracy
    best_epoch = val_accuracies.index(max(val_accuracies))
    
    # Save the model (handle both DataParallel and regular models)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch': best_epoch + 1,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': max(val_accuracies),
    }, f"{args.output_dir}/models/{config_id}.pth")
    
    # Clear memory after training
    if torch.cuda.is_available():
        del model
        del optimizer
        torch.cuda.empty_cache()
    
    return results

def run_focused_experiments(args):
    """Run experiments with different learning rates and mixed precision settings."""
    # Define configurations to test
    learning_rates = [0.01, 0.001, 0.0001]
    mixed_precision_settings = [False, True]
    
    print(f"Running {len(learning_rates) * len(mixed_precision_settings)} experiments for EfficientNet-B3")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Run each configuration
    for lr in learning_rates:
        for mp in mixed_precision_settings:
            print(f"\n{'='*80}")
            print(f"Configuration: Learning Rate = {lr}, Mixed Precision = {mp}")
            print(f"{'='*80}")
            
            try:
                # Train and evaluate with this configuration
                result = train_and_evaluate(lr, mp, args)
                all_results.append(result)
                
                # Save results so far to prevent losing progress if interrupted
                with open(f"{args.output_dir}/results_snapshot.json", 'w') as f:
                    json.dump(all_results, f, indent=4)
                
            except Exception as e:
                print(f"Error with configuration (LR={lr}, MP={mp}): {e}")
                # Save information about the failed configuration
                with open(f"{args.output_dir}/failed_configs.txt", 'a') as f:
                    f.write(f"LR={lr}, MP={mp}: {str(e)}\n")
    
    # Save final results
    with open(f"{args.output_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    return all_results

def analyze_results(results, args):
    """Analyze the results and generate plots."""
    # Create a directory for plots
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Handle empty results
    if not results:
        print("No successful experiments to analyze.")
        
        # Create a placeholder plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No successful experiments to analyze", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'no_results.png'))
        plt.close()
        
        return pd.DataFrame()
    
    # Extract key performance metrics
    performance_data = []
    for result in results:
        metrics = result['metrics']
        
        # Create a row for this configuration
        row = {
            'learning_rate': result['learning_rate'],
            'mixed_precision': 'Enabled' if result['mixed_precision'] else 'Disabled',
            'best_val_acc': metrics['best_val_acc'],
            'final_val_acc': metrics['final_val_acc'],
            'final_train_acc': metrics['final_train_acc'],
            'avg_epoch_time': metrics['avg_epoch_time'],
            'total_time': metrics['total_time']
        }
        
        performance_data.append(row)
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Save summary to CSV
    performance_df.to_csv(os.path.join(args.output_dir, "performance_summary.csv"), index=False)
    
    # Sort by validation accuracy
    top_configs = performance_df.sort_values('best_val_acc', ascending=False)
    
    print("\nConfigurations Ranked by Validation Accuracy:")
    print(top_configs[['learning_rate', 'mixed_precision', 'best_val_acc', 'avg_epoch_time']])
    
    # Create plots
    
    # 1. Learning rate comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='learning_rate', y='best_val_acc', data=performance_df)
    plt.title(f'Validation Accuracy by Learning Rate (EfficientNet-B3, Batch Size={args.batch_size})')
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', xytext = (0, 5),
                    textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'learning_rate_comparison.png'))
    plt.close()
    
    # 2. Mixed precision comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='mixed_precision', y='best_val_acc', data=performance_df)
    plt.title(f'Validation Accuracy by Mixed Precision (EfficientNet-B3, Batch Size={args.batch_size})')
    plt.xlabel('Mixed Precision')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', xytext = (0, 5),
                    textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mixed_precision_comparison.png'))
    plt.close()
    
    # 3. Training time by mixed precision
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='mixed_precision', y='avg_epoch_time', data=performance_df)
    plt.title(f'Average Training Time per Epoch by Mixed Precision (EfficientNet-B3, Batch Size={args.batch_size})')
    plt.xlabel('Mixed Precision')
    plt.ylabel('Average Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}s', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', xytext = (0, 5),
                    textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mixed_precision_time_comparison.png'))
    plt.close()
    
    # 4. Learning rate and mixed precision interaction
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='learning_rate', y='best_val_acc', hue='mixed_precision', data=performance_df)
    plt.title(f'Validation Accuracy: Learning Rate and Mixed Precision (EfficientNet-B3, Batch Size={args.batch_size})')
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', xytext = (0, 5),
                    textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'lr_mp_interaction.png'))
    plt.close()
    
    # 5. Speed vs. Accuracy scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='avg_epoch_time', y='best_val_acc', 
        hue='learning_rate', style='mixed_precision',
        s=100, data=performance_df
    )
    plt.title(f'Speed vs. Accuracy Trade-off (EfficientNet-B3, Batch Size={args.batch_size})')
    plt.xlabel('Average Epoch Time (seconds)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for each point
    for i, row in performance_df.iterrows():
        plt.annotate(
            f"LR={row['learning_rate']},\nMP={row['mixed_precision']}",
            (row['avg_epoch_time'], row['best_val_acc']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'speed_vs_accuracy.png'))
    plt.close()
    
    # 6. Learning curves for the best configuration (if any successful runs)
    if not top_configs.empty:
        best_config = top_configs.iloc[0]
        
        # Create config ID for the best model
        best_config_id = f"lr{best_config['learning_rate']}_{'mp' if best_config['mixed_precision'] == 'Enabled' else 'fp32'}"
        
        # Load metrics for the best configuration
        best_metrics_path = os.path.join(args.output_dir, "metrics", f"{best_config_id}.csv")
        
        if os.path.exists(best_metrics_path):
            best_metrics = pd.read_csv(best_metrics_path)
            
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(best_metrics['epoch'], best_metrics['train_loss'], 'b-', label='Train Loss')
            plt.plot(best_metrics['epoch'], best_metrics['val_loss'], 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss (Best Configuration)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.plot(best_metrics['epoch'], best_metrics['train_accuracy'], 'b-', label='Train Accuracy')
            plt.plot(best_metrics['epoch'], best_metrics['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy (Best Configuration)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'best_model_learning_curve.png'))
            plt.close()
    
    print(f"Analysis complete. Plots saved to {plots_dir}")
    
    return performance_df

def main():
    args = parse_args()
    print(f"Starting EfficientNet-B3 focused optimization with Multi-GPU support")
    print(f"Batch size: {args.batch_size} per GPU, using up to {args.num_gpus} GPUs")
    
    # Run experiments
    start_time = time.time()
    results = run_focused_experiments(args)
    total_time = time.time() - start_time
    
    print(f"Experiments completed in {total_time/3600:.2f} hours")
    
    # Analyze results and generate plots
    performance_summary = analyze_results(results, args)
    
    # Print final recommendations if there are successful runs
    if not performance_summary.empty:
        best_config = performance_summary.sort_values('best_val_acc', ascending=False).iloc[0]
        
        print(f"\nBest Configuration for EfficientNet-B3:")
        print(f"   - Learning Rate: {best_config['learning_rate']}")
        print(f"   - Mixed Precision: {best_config['mixed_precision']}")
        print(f"   - Best Validation Accuracy: {best_config['best_val_acc']:.2f}%")
        print(f"   - Average Epoch Time: {best_config['avg_epoch_time']:.2f} seconds")
    else:
        print("\nNo successful configurations to report.")
    
    print(f"\nPlots have been generated and saved to '{args.output_dir}/plots/' directory")
    if not performance_summary.empty:
        print(f"Full results are available in '{args.output_dir}/performance_summary.csv'")

if __name__ == "__main__":
    main()