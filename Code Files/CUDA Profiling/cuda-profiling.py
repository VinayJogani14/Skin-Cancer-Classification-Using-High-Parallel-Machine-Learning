import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler as profiler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.models as models
import time
import numpy as np
import argparse
from tqdm import tqdm

# Import your dataset classes and utilities
from DDP import SkinDiseaseDataset, get_image_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Training with CUDA profiling and hyperparameter optimization')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default='efficientnet_b3', 
                        choices=['efficientnet_b3', 'resnet50'], 
                        help='Model architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--dataset_path', type=str, default="/scratch/panchani.d/Hpc/dataset", 
                        help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--profile', action='store_true', help='Enable CUDA profiling')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--profile_steps', type=int, default=5, help='Number of steps to profile')
    parser.add_argument('--profile_memory', action='store_true', help='Profile memory usage')
    return parser.parse_args()

def load_model(model_name, num_classes):
    if model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    return model

def get_optimizer(optimizer_name, model_params, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=0.01)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9)

def prepare_data(args):
    # Load dataset image paths
    image_paths, class_labels = get_image_paths()
    
    # Get unique class names
    unique_classes = sorted(set(class_labels))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    # Split dataset
    from sklearn.model_selection import train_test_split
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(unique_classes)

def train_epoch(model, train_loader, criterion, optimizer, device, mixed_precision=False, profiling=False, profiler_obj=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
        if profiling and profiler_obj is not None:
            profiler_obj.step()
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if mixed_precision:
            with torch.cuda.amp.autocast():
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

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def profile_memory_usage(model, sample_input, device):
    """Profile memory usage for a model with given sample input."""
    # Get current GPU memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    memory_before = torch.cuda.memory_allocated(device)
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_input)
    
    memory_forward = torch.cuda.memory_allocated(device) - memory_before
    peak_memory = torch.cuda.max_memory_allocated(device)
    
    # Create a dummy loss and backward pass to check memory during backprop
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            num_classes = model.classifier[-1].out_features
        else:
            num_classes = model.classifier.out_features
    elif hasattr(model, 'fc'):
        num_classes = model.fc.out_features
    else:
        num_classes = 10  # Default
    
    dummy_target = torch.randint(0, num_classes, (sample_input.size(0),), device=device)
    dummy_output = model(sample_input)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(dummy_output, dummy_target)
    
    # Backward pass
    memory_before_backward = torch.cuda.memory_allocated(device)
    loss.backward()
    memory_backward = torch.cuda.memory_allocated(device) - memory_before_backward
    
    # Peak memory
    peak_memory_with_backward = torch.cuda.max_memory_allocated(device)
    
    return {
        "forward_pass_memory": memory_forward / (1024 * 1024),  # Convert to MB
        "backward_pass_memory": memory_backward / (1024 * 1024),
        "peak_memory": peak_memory / (1024 * 1024),
        "peak_memory_with_backward": peak_memory_with_backward / (1024 * 1024)
    }

def train_with_profiling(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, num_classes = prepare_data(args)
    
    # Load model
    model = load_model(args.model, num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # For timing
    start_time = time.time()
    epoch_times = []
    
    # Create profiler output directory
    os.makedirs('./profiler_logs', exist_ok=True)
    
    # Set up profiler if enabled
    if args.profile:
        # More detailed profiling setup
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1,
                warmup=1,
                active=args.profile_steps,
                repeat=1
            ),
            on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
        prof.start()
        profiling_active = True
    else:
        prof = None
        profiling_active = False
    
    # Memory profiling - get sample input
    if args.profile_memory:
        # Get a sample batch for memory profiling
        for sample_input, sample_target in train_loader:
            sample_input = sample_input.to(device)
            break
            
        memory_profiles = {}
        memory_profiles[args.model] = profile_memory_usage(model, sample_input, device)
        
        print(f"Memory profile for {args.model}:")
        for key, value in memory_profiles[args.model].items():
            print(f"  {key}: {value:.2f} MB")
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            args.mixed_precision, profiling=profiling_active, profiler_obj=prof
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
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
        
    # End profiling if enabled
    if args.profile:
        # Print summary of profiling data
        prof.stop()
        print("Profiling completed. Results saved to ./profiler_logs")
        
        # Get key metrics from profiler
        print("\nProfiler Summary:")
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(table)
        
        # Export profiler data in a format that can be plotted
        key_metrics = []
        for event in prof.key_averages():
            key_metrics.append({
                'name': event.key,
                'cuda_time_total': event.cuda_time_total / 1000,  # Convert to ms
                'cpu_time_total': event.cpu_time_total / 1000,    # Convert to ms
                'self_cuda_time_total': event.self_cuda_time_total / 1000,
                'self_cpu_time_total': event.self_cpu_time_total / 1000,
                'count': event.count
            })
        
        metrics_df = pd.DataFrame(key_metrics)
        metrics_df.to_csv(f'profiler_logs/{args.model}_profiling_metrics.csv', index=False)
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Save results
    config_name = f"{args.model}_{args.optimizer}_bs{args.batch_size}_lr{args.lr}"
    if args.mixed_precision:
        config_name += "_mp"
    
    results = {
        'model': args.model,
        'optimizer': args.optimizer,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'mixed_precision': args.mixed_precision,
        'total_time': total_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'train_loss': train_losses[-1],
        'train_acc': train_accuracies[-1],
        'val_loss': val_losses[-1],
        'val_acc': val_accuracies[-1]
    }
    
    # Save metrics and timing information
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, args.epochs + 1)),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'epoch_time': epoch_times
    })
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save metrics to CSV
    metrics_df.to_csv(f'results/{config_name}_metrics.csv', index=False)
    
    # Save model checkpoint
    torch.save(model.state_dict(), f'results/{config_name}_model.pth')
    
    return results

def run_comparative_analysis():
    """Run a comparative analysis between EfficientNetB3 and ResNet50 models."""
    models = ['efficientnet_b3', 'resnet50']
    batch_sizes = [32, 64, 128]
    optimizers = ['adam']
    learning_rates = [0.001]
    mixed_precision_options = [False, True]
    
    # Define a fixed set of arguments
    base_args = argparse.Namespace(
        dataset_path="/scratch/panchani.d/Hpc/dataset",
        epochs=2,
        profile=False
    )
    
    all_results = []
    
    # Run experiments with EfficientNetB3 and ResNet50
    experiment_configs = []
    
    # Model comparison with different batch sizes
    for model in models:
        for batch_size in batch_sizes:
            experiment_configs.append({
                'model': model, 
                'batch_size': batch_size, 
                'optimizer': 'adam', 
                'lr': 0.001, 
                'mixed_precision': False
            })
    
    # Mixed precision comparison
    for model in models:
        for mp in mixed_precision_options:
            experiment_configs.append({
                'model': model, 
                'batch_size': 64, 
                'optimizer': 'adam', 
                'lr': 0.001, 
                'mixed_precision': mp
            })
    
    for config in experiment_configs:
        print(f"\n{'='*80}")
        print(f"Running experiment with config: {config}")
        print(f"{'='*80}")
        
        args = argparse.Namespace(**{**vars(base_args), **config})
        result = train_with_profiling(args)
        all_results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/model_comparison_results.csv', index=False)
    
    return results_df

def plot_cuda_profiling_results():
    """Generate enhanced plots specifically for CUDA profiling analysis."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Try to load profiling data
        efficientnet_profiling = pd.read_csv('profiler_logs/efficientnet_b3_profiling_metrics.csv')
        resnet_profiling = pd.read_csv('profiler_logs/resnet50_profiling_metrics.csv')
        
        # 1. Top operations by CUDA time - EfficientNetB3
        plt.figure(figsize=(14, 6))
        top_efficient = efficientnet_profiling.nlargest(10, 'cuda_time_total')
        sns.barplot(x='cuda_time_total', y='name', data=top_efficient)
        plt.title('Top 10 Operations by CUDA Time - EfficientNetB3')
        plt.xlabel('CUDA Time (ms)')
        plt.ylabel('Operation')
        plt.tight_layout()
        plt.savefig('plots/efficientnet_top_cuda_ops.png')
        plt.close()
        
        # 2. Top operations by CUDA time - ResNet50
        plt.figure(figsize=(14, 6))
        top_resnet = resnet_profiling.nlargest(10, 'cuda_time_total')
        sns.barplot(x='cuda_time_total', y='name', data=top_resnet)
        plt.title('Top 10 Operations by CUDA Time - ResNet50')
        plt.xlabel('CUDA Time (ms)')
        plt.ylabel('Operation')
        plt.tight_layout()
        plt.savefig('plots/resnet_top_cuda_ops.png')
        plt.close()
        
        # 3. Compare common operations between models
        common_ops = pd.merge(
            efficientnet_profiling[['name', 'cuda_time_total']].rename(columns={'cuda_time_total': 'efficientnet_time'}),
            resnet_profiling[['name', 'cuda_time_total']].rename(columns={'cuda_time_total': 'resnet_time'}),
            on='name', how='inner'
        )
        
        if not common_ops.empty:
            common_ops = common_ops.nlargest(10, ['efficientnet_time', 'resnet_time'])
            
            # Reshape for plotting
            common_ops_melted = pd.melt(
                common_ops, 
                id_vars=['name'], 
                value_vars=['efficientnet_time', 'resnet_time'],
                var_name='model', 
                value_name='cuda_time'
            )
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='cuda_time', y='name', hue='model', data=common_ops_melted)
            plt.title('Common CUDA Operations Comparison between Models')
            plt.xlabel('CUDA Time (ms)')
            plt.ylabel('Operation')
            plt.tight_layout()
            plt.savefig('plots/model_common_ops_comparison.png')
            plt.close()
        
        print("CUDA profiling plots generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate CUDA profiling plots: {e}")
        print("Run with --profile option to generate profiling data first.")

def plot_model_comparison_results():
    """Generate plots comparing EfficientNetB3 and ResNet50 performance."""
    try:
        # Load comparison results
        results_df = pd.read_csv('results/model_comparison_results.csv')
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # 1. Training time comparison by model and batch size
        plt.figure(figsize=(12, 6))
        batch_model_grouped = results_df[~results_df['mixed_precision']].groupby(['model', 'batch_size'])['avg_epoch_time'].mean().reset_index()
        batch_model_pivot = batch_model_grouped.pivot(index='batch_size', columns='model', values='avg_epoch_time')
        batch_model_pivot.plot(kind='bar', figsize=(12, 6))
        plt.title('Average Training Time per Epoch by Model and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plots/model_batch_time_comparison.png')
        plt.close()
        
        # 2. Mixed precision impact on training time
        plt.figure(figsize=(12, 6))
        mp_grouped = results_df[results_df['batch_size'] == 64].groupby(['model', 'mixed_precision'])['avg_epoch_time'].mean().reset_index()
        mp_pivot = mp_grouped.pivot(index='mixed_precision', columns='model', values='avg_epoch_time')
        mp_pivot.plot(kind='bar', figsize=(12, 6))
        plt.title('Impact of Mixed Precision on Training Time')
        plt.xlabel('Mixed Precision')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(ticks=[0, 1], labels=['Disabled', 'Enabled'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plots/mixed_precision_comparison.png')
        plt.close()
        
        # 3. Validation accuracy comparison
        plt.figure(figsize=(12, 6))
        acc_grouped = results_df.groupby(['model', 'batch_size'])['val_acc'].mean().reset_index()
        acc_pivot = acc_grouped.pivot(index='batch_size', columns='model', values='val_acc')
        acc_pivot.plot(kind='bar', figsize=(12, 6))
        plt.title('Validation Accuracy by Model and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Validation Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plots/model_accuracy_comparison.png')
        plt.close()
        
        # 4. Efficiency comparison (Training time vs Validation accuracy)
        plt.figure(figsize=(12, 6))
        results_df['efficiency'] = results_df['val_acc'] / results_df['avg_epoch_time']
        eff_grouped = results_df.groupby(['model', 'batch_size'])['efficiency'].mean().reset_index()
        eff_pivot = eff_grouped.pivot(index='batch_size', columns='model', values='efficiency')
        eff_pivot.plot(kind='bar', figsize=(12, 6))
        plt.title('Training Efficiency (Accuracy per Time) by Model and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Efficiency (Accuracy % / Second)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plots/model_efficiency_comparison.png')
        plt.close()
        
        print("Model comparison plots generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate model comparison plots: {e}")
        print("Run comparative analysis first to generate data.")

def generate_hardware_utilization_plots():
    """Generate plots showing hardware utilization during training."""
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Get GPU properties
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            
            # Sample data - in a real scenario, you would collect this during training
            # For demonstration purposes, we'll create sample utilization profiles
            
            # 1. GPU Memory Utilization over time
            plt.figure(figsize=(12, 6))
            
            # Sample data - Replace with actual measurements
            time_points = np.arange(0, 100, 5)
            
            # EfficientNetB3 memory profile (synthetic data)
            memory_profile_efficient = 2000 + 4000 * (1 - np.exp(-time_points/20))
            # ResNet50 memory profile (synthetic data)
            memory_profile_resnet = 1500 + 3500 * (1 - np.exp(-time_points/15))
            
            plt.plot(time_points, memory_profile_efficient, 'b-', label='EfficientNetB3')
            plt.plot(time_points, memory_profile_resnet, 'r-', label='ResNet50')
            
            plt.xlabel('Training Progress (iterations)')
            plt.ylabel('GPU Memory Usage (MB)')
            plt.title('GPU Memory Utilization During Training')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/gpu_memory_utilization.png')
            plt.close()
            
            # 2. GPU Compute Utilization
            plt.figure(figsize=(12, 6))
            
            # Sample data - Replace with actual measurements
            # GPU utilization percentage over time for each model
            util_efficient = 75 + 20 * np.sin(time_points/10)
            util_efficient = np.clip(util_efficient, 0, 100)
            
            util_resnet = 85 + 15 * np.sin(time_points/8)
            util_resnet = np.clip(util_resnet, 0, 100)
            
            plt.plot(time_points, util_efficient, 'b-', label='EfficientNetB3')
            plt.plot(time_points, util_resnet, 'r-', label='ResNet50')
            
            plt.xlabel('Training Progress (iterations)')
            plt.ylabel('GPU Utilization (%)')
            plt.title('GPU Compute Utilization During Training')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/gpu_compute_utilization.png')
            plt.close()
            
            # 3. Operation Distribution Plot - EfficientNetB3
            # This is a substitute for the pie chart with more meaningful data
            plt.figure(figsize=(12, 7))
            
            # Sample data - Replace with actual measurements
            operations = ['Conv2D', 'BatchNorm', 'ReLU', 'MaxPool', 
                        'AvgPool', 'Linear', 'Softmax', 'DataLoader', 'Optimizer']
            
            # Time percentage for each operation (synthetic data)
            efficient_times = [45, 10, 5, 7, 3, 8, 2, 15, 5]
            resnet_times = [40, 12, 6, 9, 0, 10, 3, 15, 5]
            
            # Create grouped bar chart
            x = np.arange(len(operations))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 7))
            rects1 = ax.bar(x - width/2, efficient_times, width, label='EfficientNetB3')
            rects2 = ax.bar(x + width/2, resnet_times, width, label='ResNet50')
            
            ax.set_ylabel('Time Percentage (%)')
            ax.set_title('Time Distribution by Operation Type')
            ax.set_xticks(x)
            ax.set_xticklabels(operations)
            ax.legend()
            
            # Add labels on top of bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height}%',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('plots/operation_time_distribution.png')
            plt.close()
            
            print("Hardware utilization plots generated successfully.")
        else:
            print("No GPUs available for hardware utilization plots.")
    except Exception as e:
        print(f"Warning: Could not generate hardware utilization plots: {e}")

def analyze_kernel_launches():
    """Generate plots analyzing CUDA kernel launches and execution patterns."""
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Sample kernel data - in a real scenario, this would come from profiling
        kernel_data = {
            'efficientnet_b3': {
                'kernel_names': ['matmul', 'conv2d', 'relu', 'norm', 'pool', 'other'],
                'launch_counts': [320, 480, 560, 230, 120, 290],
                'avg_duration_ms': [0.45, 0.85, 0.12, 0.22, 0.18, 0.30]
            },
            'resnet50': {
                'kernel_names': ['matmul', 'conv2d', 'relu', 'norm', 'pool', 'other'],
                'launch_counts': [280, 520, 510, 250, 150, 310],
                'avg_duration_ms': [0.42, 0.92, 0.10, 0.25, 0.15, 0.32]
            }
        }
        
        # 1. Kernel Launch Count Comparison
        plt.figure(figsize=(14, 7))
        width = 0.35
        x = np.arange(len(kernel_data['efficientnet_b3']['kernel_names']))
        
        fig, ax = plt.subplots(figsize=(14, 7))
        rects1 = ax.bar(x - width/2, kernel_data['efficientnet_b3']['launch_counts'], width, label='EfficientNetB3')
        rects2 = ax.bar(x + width/2, kernel_data['resnet50']['launch_counts'], width, label='ResNet50')
        
        ax.set_ylabel('Number of Kernel Launches')
        ax.set_title('CUDA Kernel Launch Count by Operation Type')
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_data['efficientnet_b3']['kernel_names'])
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/kernel_launch_count.png')
        plt.close()
        
        # 2. Average Kernel Duration
        plt.figure(figsize=(14, 7))
        
        fig, ax = plt.subplots(figsize=(14, 7))
        rects1 = ax.bar(x - width/2, kernel_data['efficientnet_b3']['avg_duration_ms'], width, label='EfficientNetB3')
        rects2 = ax.bar(x + width/2, kernel_data['resnet50']['avg_duration_ms'], width, label='ResNet50')
        
        ax.set_ylabel('Average Duration (ms)')
        ax.set_title('Average CUDA Kernel Duration by Operation Type')
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_data['efficientnet_b3']['kernel_names'])
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/kernel_avg_duration.png')
        plt.close()
        
        # 3. Total Kernel Time (Launch Count * Avg Duration)
        plt.figure(figsize=(14, 7))
        
        efficientnet_total_time = [count * duration for count, duration in 
                                 zip(kernel_data['efficientnet_b3']['launch_counts'], 
                                     kernel_data['efficientnet_b3']['avg_duration_ms'])]
        
        resnet_total_time = [count * duration for count, duration in 
                           zip(kernel_data['resnet50']['launch_counts'], 
                               kernel_data['resnet50']['avg_duration_ms'])]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        rects1 = ax.bar(x - width/2, efficientnet_total_time, width, label='EfficientNetB3')
        rects2 = ax.bar(x + width/2, resnet_total_time, width, label='ResNet50')
        
        ax.set_ylabel('Total Kernel Time (ms)')
        ax.set_title('Total CUDA Kernel Time by Operation Type')
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_data['efficientnet_b3']['kernel_names'])
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/kernel_total_time.png')
        plt.close()
        
        print("Kernel analysis plots generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate kernel analysis plots: {e}")

def generate_memory_footprint_plots():
    """Generate plots showing memory footprint of different models and batch sizes."""
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Sample memory data for both models at different batch sizes
        # In a real scenario, this would come from actual profiling
        batch_sizes = [16, 32, 64, 128, 256]
        
        # Memory usage in MB for different components (forward pass)
        memory_data = {
            'efficientnet_b3': {
                'model_params': [180] * 5,  # Same for all batch sizes
                'forward_activations': [350, 700, 1400, 2800, 5600],
                'backward_gradients': [420, 840, 1680, 3360, 6720],
                'optimizer_state': [100] * 5  # Same for all batch sizes
            },
            'resnet50': {
                'model_params': [150] * 5,  # Same for all batch sizes
                'forward_activations': [300, 600, 1200, 2400, 4800],
                'backward_gradients': [360, 720, 1440, 2880, 5760],
                'optimizer_state': [80] * 5  # Same for all batch sizes
            }
        }
        
        # 1. Memory usage breakdown by component for a fixed batch size (64)
        plt.figure(figsize=(14, 8))
        
        # Get the index of batch size 64
        bs_idx = batch_sizes.index(64)
        
        # Create data for stacked bar chart
        models = ['EfficientNetB3', 'ResNet50']
        components = ['Model Parameters', 'Forward Activations', 'Backward Gradients', 'Optimizer State']
        
        efficient_memory = [
            memory_data['efficientnet_b3']['model_params'][bs_idx],
            memory_data['efficientnet_b3']['forward_activations'][bs_idx],
            memory_data['efficientnet_b3']['backward_gradients'][bs_idx],
            memory_data['efficientnet_b3']['optimizer_state'][bs_idx]
        ]
        
        resnet_memory = [
            memory_data['resnet50']['model_params'][bs_idx],
            memory_data['resnet50']['forward_activations'][bs_idx],
            memory_data['resnet50']['backward_gradients'][bs_idx],
            memory_data['resnet50']['optimizer_state'][bs_idx]
        ]
        
        x = np.arange(len(models))
        width = 0.6
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bottom_efficient = 0
        bottom_resnet = 0
        
        bars = []
        for i, component in enumerate(components):
            efficient_bar = ax.bar(0, efficient_memory[i], width, bottom=bottom_efficient, label=component if i == 0 else "")
            resnet_bar = ax.bar(1, resnet_memory[i], width, bottom=bottom_resnet, label="" if i > 0 else "")
            
            bottom_efficient += efficient_memory[i]
            bottom_resnet += resnet_memory[i]
            
            bars.append((efficient_bar, resnet_bar))
        
        # Add labels at the center of each segment
        for i, ((efficient_bar, resnet_bar), component) in enumerate(zip(bars, components)):
            efficient_rect = efficient_bar[0]
            resnet_rect = resnet_bar[0]
            
            height_efficient = efficient_rect.get_height()
            height_resnet = resnet_rect.get_height()
            
            if height_efficient > 100:  # Only label if segment is large enough
                ax.text(efficient_rect.get_x() + efficient_rect.get_width()/2,
                        efficient_rect.get_y() + height_efficient/2,
                        f'{component}\n{height_efficient} MB',
                        ha='center', va='center', color='white', fontweight='bold')
            
            if height_resnet > 100:  # Only label if segment is large enough
                ax.text(resnet_rect.get_x() + resnet_rect.get_width()/2,
                        resnet_rect.get_y() + height_resnet/2,
                        f'{component}\n{height_resnet} MB',
                        ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title(f'GPU Memory Usage Breakdown (Batch Size: 64)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        
        # Add total memory usage on top of each bar
        ax.text(0, bottom_efficient + 100, f'Total: {bottom_efficient} MB', 
                ha='center', va='bottom', fontweight='bold')
        ax.text(1, bottom_resnet + 100, f'Total: {bottom_resnet} MB', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(components))
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/memory_breakdown_bs64.png')
        plt.close()
        
        # 2. Memory scaling with batch size
        plt.figure(figsize=(14, 7))
        
        # Calculate total memory for each model at each batch size
        efficientnet_total = [sum(values) for values in zip(
            memory_data['efficientnet_b3']['model_params'],
            memory_data['efficientnet_b3']['forward_activations'],
            memory_data['efficientnet_b3']['backward_gradients'],
            memory_data['efficientnet_b3']['optimizer_state']
        )]
        
        resnet_total = [sum(values) for values in zip(
            memory_data['resnet50']['model_params'],
            memory_data['resnet50']['forward_activations'],
            memory_data['resnet50']['backward_gradients'],
            memory_data['resnet50']['optimizer_state']
        )]
        
        plt.plot(batch_sizes, efficientnet_total, 'bo-', linewidth=2, label='EfficientNetB3')
        plt.plot(batch_sizes, resnet_total, 'ro-', linewidth=2, label='ResNet50')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Total Memory Usage (MB)')
        plt.title('GPU Memory Usage vs Batch Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add horizontal line for typical GPU memory limit (e.g., 16GB)
        plt.axhline(y=16000, color='gray', linestyle='--', label='16GB GPU Memory Limit')
        plt.text(batch_sizes[0], 16200, '16GB VRAM Limit', va='bottom', ha='left')
        
        plt.tight_layout()
        plt.savefig('plots/memory_vs_batch_size.png')
        plt.close()
        
        print("Memory footprint plots generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate memory footprint plots: {e}")

def main():
    parser = argparse.ArgumentParser(description='CUDA Profiling and Analysis for EfficientNetB3 and ResNet50')
    parser.add_argument('--run_training', action='store_true', help='Run model training with profiling')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default='efficientnet_b3', 
                      choices=['efficientnet_b3', 'resnet50'], help='Model architecture')
    parser.add_argument('--profile', action='store_true', help='Enable detailed CUDA profiling')
    parser.add_argument('--profile_memory', action='store_true', help='Profile memory usage')
    parser.add_argument('--compare_models', action='store_true', help='Run comparative analysis between models')
    parser.add_argument('--generate_plots', action='store_true', help='Generate all analysis plots')
    
    args = parser.parse_args()
    
    print("CUDA Profiling and Analysis for EfficientNetB3 and ResNet50")
    print("-" * 60)
    
    if args.run_training:
        # Run single model training with profiling
        training_args = parse_args()
        training_args.model = args.model
        training_args.batch_size = args.batch_size
        training_args.profile = args.profile
        training_args.profile_memory = args.profile_memory
        
        results = train_with_profiling(training_args)
        print(f"Training completed in {results['total_time']:.2f} seconds")
        print(f"Validation accuracy: {results['val_acc']:.2f}%")
    
    if args.compare_models:
        # Run comparative analysis between models
        print("\nRunning comparative analysis between EfficientNetB3 and ResNet50...")
        results_df = run_comparative_analysis()
        print("Comparative analysis completed.")
    
    if args.generate_plots or not (args.run_training or args.compare_models):
        # Generate all analysis plots
        print("\nGenerating analysis plots...")
        
        # CUDA profiling plots
        plot_cuda_profiling_results()
        
        # Model comparison plots
        plot_model_comparison_results()
        
        # Hardware utilization plots
        generate_hardware_utilization_plots()
        
        # Kernel analysis plots
        analyze_kernel_launches()
        
        # Memory footprint plots
        generate_memory_footprint_plots()
        
        print("All analysis plots generated.")

if __name__ == "__main__":
    main()