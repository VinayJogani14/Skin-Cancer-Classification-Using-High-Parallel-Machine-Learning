import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm
import socket
import platform
import json
from sklearn.model_selection import train_test_split

# Import your dataset classes and utilities
from DDP import SkinDiseaseDataset, get_image_paths
import torchvision.models as models

def parse_args():
    parser = argparse.ArgumentParser(description='GPU Scaling Comparison for EfficientNet-B3')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--output_dir', type=str, default="AMP-plots", 
                        help='Directory to save results')
    parser.add_argument('--iterations', type=int, default=20, 
                        help='Number of iterations to run for each test')
    parser.add_argument('--include_cpu', action='store_true',
                        help='Include CPU in the benchmarks (much slower)')
    return parser.parse_args()

def get_system_info():
    """Get basic information about the system."""
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__
    }
    
    # CPU information
    info['cpu_model'] = platform.processor()
    info['cpu_count'] = os.cpu_count()
    
    # GPU information
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_models'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        info['cuda_available'] = False
    
    return info

def prepare_data(batch_size):
    """Prepare data loaders for performance testing."""
    # Load dataset image paths
    image_paths, class_labels = get_image_paths()
    
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    train_dataset = SkinDiseaseDataset(
        image_paths=train_image_paths, 
        class_labels=train_labels,
        class_to_idx=class_to_idx,
        transform=train_transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, len(unique_classes)

def load_efficientnet_b3(num_classes):
    """Load EfficientNet-B3 model."""
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

def benchmark_training(model, data_loader, device_config, iterations, mixed_precision=False):
    """Benchmark training performance with specified device config."""
    # device_config can be:
    # - 'cpu': Use CPU
    # - 'gpu:0': Use single GPU
    # - 'gpu:0,1': Use 2 GPUs
    # - 'gpu:0,1,2,3': Use 4 GPUs
    
    # Clear CUDA cache to get accurate measurements
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Setup the device(s)
    if device_config == 'cpu':
        device = torch.device('cpu')
        model = model.to(device)
    else:
        gpu_ids = [int(i) for i in device_config.split(':')[1].split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        
        if len(gpu_ids) > 1:
            print(f"  Using DataParallel with GPUs: {gpu_ids}")
            model = nn.DataParallel(model, device_ids=gpu_ids)
        
        model = model.to(device)
    
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create a scaler for automatic mixed precision
    scaler = torch.amp.GradScaler() if mixed_precision else None
    
    # Get a batch for testing
    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        
        if mixed_precision:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Synchronize to ensure timing is correct
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time the full iteration
    iteration_times = []
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        
        if mixed_precision:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Synchronize before recording the end time
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        iteration_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(iteration_times)
    min_time = np.min(iteration_times)
    max_time = np.max(iteration_times)
    std_time = np.std(iteration_times)
    
    # Calculate throughput (images/second)
    effective_batch_size = data_loader.batch_size * (len(gpu_ids) if device_config != 'cpu' and len(gpu_ids) > 1 else 1)
    throughput = effective_batch_size / avg_time
    
    # Get peak memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_allocated = 0
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'throughput': throughput,
        'memory_mb': memory_allocated,
        'effective_batch_size': effective_batch_size
    }

def run_scaling_benchmarks(args):
    """Run benchmarks with different GPU configurations."""
    # Get system information
    system_info = get_system_info()
    print("System Information:")
    print(f"  Hostname: {system_info['hostname']}")
    print(f"  Platform: {system_info['platform']}")
    print(f"  PyTorch: {system_info['torch_version']}")
    
    if torch.cuda.is_available():
        print(f"  GPU Count: {system_info['gpu_count']}")
        for i, gpu in enumerate(system_info['gpu_models']):
            print(f"  GPU {i}: {gpu}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save system information
    with open(os.path.join(args.output_dir, 'system_info.json'), 'w') as f:
        json.dump(system_info, f, indent=4)
    
    # Prepare data loader
    data_loader, num_classes = prepare_data(args.batch_size)
    
    # Define configurations to test
    configs = []
    
    # Include CPU if requested
    if args.include_cpu:
        configs.append('cpu')
    
    # Add GPU configurations
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        
        # Single GPU
        configs.append('gpu:0')
        
        # 2 GPUs (if available)
        if gpu_count >= 2:
            configs.append('gpu:0,1')
        
        # 4 GPUs (if available)
        if gpu_count >= 4:
            configs.append('gpu:0,1,2,3')
    
    # Load model
    model = load_efficientnet_b3(num_classes)
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024 * 1024)  # Size in MB (assuming float32)
    
    print(f"\nEfficientNet-B3 model has {param_count:,} parameters ({model_size_mb:.2f} MB)")
    
    # Results container
    results = []
    
    # Run benchmarks for each configuration
    for config in configs:
        config_name = config
        if config == 'cpu':
            print(f"\nBenchmarking on CPU...")
            device_name = "CPU"
        else:
            gpu_ids = [int(i) for i in config.split(':')[1].split(',')]
            if len(gpu_ids) == 1:
                print(f"\nBenchmarking on single GPU (GPU {gpu_ids[0]})...")
                device_name = f"1 GPU"
            else:
                print(f"\nBenchmarking on {len(gpu_ids)} GPUs (GPUs {gpu_ids})...")
                device_name = f"{len(gpu_ids)} GPUs"
        
        # Standard precision training
        print(f"  Running standard precision training...")
        
        try:
            train_results = benchmark_training(
                model.cpu() if config == 'cpu' else model, 
                data_loader, 
                config, 
                args.iterations,
                mixed_precision=False
            )
            
            results.append({
                'config': config_name,
                'device': device_name,
                'precision': 'FP32',
                'batch_size': args.batch_size,
                'effective_batch_size': train_results['effective_batch_size'],
                'avg_time': train_results['avg_time'],
                'min_time': train_results['min_time'],
                'max_time': train_results['max_time'],
                'std_time': train_results['std_time'],
                'throughput': train_results['throughput'],
                'memory_mb': train_results.get('memory_mb', 0)
            })
            
            print(f"    Average time: {train_results['avg_time']*1000:.2f} ms")
            print(f"    Throughput: {train_results['throughput']:.2f} images/second")
            print(f"    Effective batch size: {train_results['effective_batch_size']}")
            if config != 'cpu':
                print(f"    Memory used: {train_results['memory_mb']:.2f} MB")
        
        except Exception as e:
            print(f"    Error: {e}")
        
        # Mixed precision training (only for GPUs)
        if config != 'cpu':
            print(f"  Running mixed precision training...")
            
            try:
                train_mp_results = benchmark_training(
                    model, 
                    data_loader, 
                    config, 
                    args.iterations,
                    mixed_precision=True
                )
                
                results.append({
                    'config': config_name,
                    'device': device_name,
                    'precision': 'Mixed Precision',
                    'batch_size': args.batch_size,
                    'effective_batch_size': train_mp_results['effective_batch_size'],
                    'avg_time': train_mp_results['avg_time'],
                    'min_time': train_mp_results['min_time'],
                    'max_time': train_mp_results['max_time'],
                    'std_time': train_mp_results['std_time'],
                    'throughput': train_mp_results['throughput'],
                    'memory_mb': train_mp_results.get('memory_mb', 0)
                })
                
                print(f"    Average time: {train_mp_results['avg_time']*1000:.2f} ms")
                print(f"    Throughput: {train_mp_results['throughput']:.2f} images/second")
                print(f"    Effective batch size: {train_mp_results['effective_batch_size']}")
                print(f"    Memory used: {train_mp_results['memory_mb']:.2f} MB")
            
            except Exception as e:
                print(f"    Error: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, 'scaling_benchmark_results.csv'), index=False)
    
    # Generate plots
    generate_scaling_plots(results_df, args)
    
    return results_df

def generate_scaling_plots(results_df, args):
    """Generate scaling comparison plots."""
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Throughput comparison across devices
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='device', y='throughput', hue='precision', data=results_df)
    plt.title(f'EfficientNet-B3: Training Throughput by Device (Batch Size {args.batch_size})')
    plt.xlabel('Device')
    plt.ylabel('Throughput (images/second)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', xytext = (0, 5),
                    textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'throughput_comparison.png'))
    plt.close()
    
    # 2. Speedup relative to CPU (if CPU is included)
    if 'cpu' in results_df['config'].values:
        # Get CPU throughput as baseline
        cpu_throughput = results_df[results_df['config'] == 'cpu']['throughput'].values[0]
        
        # Calculate speedup
        speedup_df = results_df.copy()
        speedup_df['speedup'] = speedup_df['throughput'] / cpu_throughput
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='device', y='speedup', hue='precision', data=speedup_df)
        plt.title(f'EfficientNet-B3: Training Speedup Relative to CPU (Batch Size {args.batch_size})')
        plt.xlabel('Device')
        plt.ylabel('Speedup (x)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of the bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.1f}x', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'bottom', xytext = (0, 5),
                        textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cpu_relative_speedup.png'))
        plt.close()
    
    # 3. GPU scaling efficiency (relative to single GPU)
    gpu_results = results_df[results_df['config'] != 'cpu'].copy()
    
    if not gpu_results.empty and len(gpu_results['device'].unique()) > 1:
        # Get results for different GPU counts
        gpu_counts = []
        for device in gpu_results['device'].unique():
            if 'GPUs' in device:
                count = int(device.split(' ')[0])
                gpu_counts.append(count)
        
        if len(gpu_counts) > 1:
            # Separate standard and mixed precision results
            for precision in gpu_results['precision'].unique():
                precision_results = gpu_results[gpu_results['precision'] == precision].copy()
                
                if len(precision_results) >= 2:  # Need at least 2 data points
                    # Get single GPU throughput
                    single_gpu_throughput = precision_results[precision_results['device'] == '1 GPU']['throughput'].values[0]
                    
                    # Create data for plot
                    gpu_counts_plot = []
                    throughputs_plot = []
                    for device in precision_results['device'].unique():
                        if 'GPU' in device:
                            if 'GPUs' in device:
                                count = int(device.split(' ')[0])
                            else:
                                count = 1
                            throughput = precision_results[precision_results['device'] == device]['throughput'].values[0]
                            gpu_counts_plot.append(count)
                            throughputs_plot.append(throughput)
                    
                    # Sort by GPU count
                    gpu_counts_plot, throughputs_plot = zip(*sorted(zip(gpu_counts_plot, throughputs_plot)))
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(gpu_counts_plot, throughputs_plot, 'o-', linewidth=2, label='Actual')
                    
                    # Ideal scaling line
                    ideal_throughputs = [single_gpu_throughput * count for count in gpu_counts_plot]
                    plt.plot(gpu_counts_plot, ideal_throughputs, 'r--', linewidth=2, label='Ideal Linear Scaling')
                    
                    plt.title(f'EfficientNet-B3: {precision} Training Throughput Scaling with GPU Count')
                    plt.xlabel('Number of GPUs')
                    plt.ylabel('Throughput (images/second)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    
                    # Add data labels
                    for i, (x, y) in enumerate(zip(gpu_counts_plot, throughputs_plot)):
                        plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'gpu_scaling_{precision.replace(" ", "_").lower()}.png'))
                    plt.close()
                    
                    # Calculate scaling efficiency
                    plt.figure(figsize=(10, 6))
                    efficiencies = [throughputs_plot[i] / (ideal_throughputs[i]) * 100 for i in range(len(gpu_counts_plot))]
                    
                    plt.bar(gpu_counts_plot, efficiencies, width=0.6)
                    plt.axhline(y=100, color='r', linestyle='--', label='Ideal (100%)')
                    
                    plt.title(f'EfficientNet-B3: {precision} GPU Scaling Efficiency')
                    plt.xlabel('Number of GPUs')
                    plt.ylabel('Scaling Efficiency (%)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add percentage labels
                    for i, (x, y) in enumerate(zip(gpu_counts_plot, efficiencies)):
                        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'gpu_efficiency_{precision.replace(" ", "_").lower()}.png'))
                    plt.close()
    
    # 4. Mixed precision speedup for each device
    if 'Mixed Precision' in results_df['precision'].values and 'FP32' in results_df['precision'].values:
        # Calculate the speedup from mixed precision for each device
        devices = []
        speedups = []
        
        for device in results_df['device'].unique():
            if 'CPU' in device:
                continue  # Skip CPU for mixed precision comparison
                
            device_data = results_df[results_df['device'] == device]
            std_data = device_data[device_data['precision'] == 'FP32']
            amp_data = device_data[device_data['precision'] == 'Mixed Precision']
            
            if not std_data.empty and not amp_data.empty:
                devices.append(device)
                speedups.append(amp_data['throughput'].iloc[0] / std_data['throughput'].iloc[0])
        
        if devices:
            plt.figure(figsize=(10, 6))
            ax = plt.bar(devices, speedups, width=0.6)
            plt.title('EfficientNet-B3: Mixed Precision Training Speedup by Device')
            plt.xlabel('Device')
            plt.ylabel('Speedup (Mixed Precision / FP32)')
            plt.axhline(y=1.0, color='r', linestyle='--', label='No Speedup')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add speedup labels
            for i, v in enumerate(speedups):
                plt.text(i, v + 0.02, f"{v:.2f}x", ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'mixed_precision_speedup.png'))
            plt.close()
    
    # 5. Memory usage comparison
    if 'memory_mb' in results_df.columns:
        gpu_memory_df = results_df[results_df['config'] != 'cpu'].copy()
        
        if not gpu_memory_df.empty:
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='device', y='memory_mb', hue='precision', data=gpu_memory_df)
            plt.title(f'EfficientNet-B3: GPU Memory Usage by Configuration (Batch Size {args.batch_size})')
            plt.xlabel('Device')
            plt.ylabel('Memory Usage (MB)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'bottom', xytext = (0, 5),
                            textcoords = 'offset points')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'memory_usage.png'))
            plt.close()
            
            # 6. Memory efficiency (throughput per MB)
            memory_eff = gpu_memory_df.copy()
            # Avoid division by zero
            memory_eff['memory_mb'] = memory_eff['memory_mb'].apply(lambda x: max(x, 1))
            memory_eff['efficiency'] = memory_eff['throughput'] / memory_eff['memory_mb']
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='device', y='efficiency', hue='precision', data=memory_eff)
            plt.title('EfficientNet-B3: Memory Efficiency (Throughput per MB)')
            plt.xlabel('Device')
            plt.ylabel('Efficiency (images/second/MB)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_height():.4f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'bottom', xytext = (0, 5),
                            textcoords = 'offset points',
                            fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'memory_efficiency.png'))
            plt.close()
    
    print(f"Scaling comparison plots saved to {plots_dir}")

def main():
    args = parse_args()
    print(f"Running GPU Scaling Comparison for EfficientNet-B3")
    print(f"Batch size: {args.batch_size}, Iterations: {args.iterations}")
    
    # Run benchmarks
    results_df = run_scaling_benchmarks(args)
    
    print(f"GPU scaling comparison completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()