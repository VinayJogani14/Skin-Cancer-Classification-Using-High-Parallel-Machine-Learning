import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path

def generate_training_metrics_plots():
    """Generate plots from the training metrics."""
    print("Generating training metrics plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load training metrics
        metrics_df = pd.read_csv('training_metrics.csv')
        
        if not metrics_df.empty:
            # Plot loss curves
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-', label='Training Loss')
            plt.plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.plot(metrics_df['epoch'], metrics_df['train_accuracy'], 'b-', label='Training Accuracy')
            plt.plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('plots/training_metrics.png')
            plt.close()
            
            print("Training metrics plots saved to 'plots/training_metrics.png'")
        else:
            print("No training metrics data found.")
    except Exception as e:
        print(f"Error generating training metrics plots: {e}")

def generate_ddp_comparison_plots():
    """Generate comparison plots for DDP scaling."""
    print("Generating DDP comparison plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load DDP timing data
        ddp_data = pd.read_csv('training_times.csv')
        
        if not ddp_data.empty:
            # Plot training time vs number of GPUs
            plt.figure(figsize=(12, 6))
            
            # Group by batch size
            for bs in ddp_data['Batch Size'].unique():
                bs_data = ddp_data[ddp_data['Batch Size'] == bs]
                plt.plot(bs_data['GPUs'], bs_data['Training Time (Seconds)'], 
                         marker='o', linewidth=2, label=f'Batch Size {bs}')
            
            # Add ideal scaling line if we have single-GPU data
            if 1 in ddp_data['GPUs'].values:
                # Get the single-GPU training time for each batch size
                gpu_times = {}
                for bs in ddp_data['Batch Size'].unique():
                    bs_data = ddp_data[ddp_data['Batch Size'] == bs]
                    if 1 in bs_data['GPUs'].values:
                        gpu_times[bs] = bs_data[bs_data['GPUs'] == 1]['Training Time (Seconds)'].values[0]
                
                # Plot ideal scaling for each batch size
                max_gpus = ddp_data['GPUs'].max()
                gpu_range = range(1, max_gpus + 1)
                
                for bs, time in gpu_times.items():
                    ideal_times = [time / g for g in gpu_range]
                    plt.plot(gpu_range, ideal_times, '--', alpha=0.5, label=f'Ideal Scaling (BS {bs})')
            
            plt.xlabel('Number of GPUs')
            plt.ylabel('Training Time (seconds)')
            plt.title('DDP Training Time vs. Number of GPUs')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig('plots/ddp_training_time_comparison.png')
            plt.close()
            
            # Plot speedup vs number of GPUs
            plt.figure(figsize=(12, 6))
            
            for bs in ddp_data['Batch Size'].unique():
                bs_data = ddp_data[ddp_data['Batch Size'] == bs]
                if 1 in bs_data['GPUs'].values:
                    single_gpu_time = bs_data[bs_data['GPUs'] == bs_data['GPUs'].min()]['Training Time (Seconds)'].values[0]
                    bs_data = bs_data.copy()
                    bs_data['Speedup'] = single_gpu_time / bs_data['Training Time (Seconds)']
                    plt.plot(bs_data['GPUs'], bs_data['Speedup'], marker='o', linewidth=2, label=f'Batch Size {bs}')
            
            # Add ideal speedup line
            max_gpus = ddp_data['GPUs'].max()
            gpu_range = range(1, max_gpus + 1)
            ideal_speedup = [g for g in gpu_range]
            plt.plot(gpu_range, ideal_speedup, 'k--', label='Ideal Scaling')
            
            plt.xlabel('Number of GPUs')
            plt.ylabel('Speedup (x)')
            plt.title('DDP Training Speedup vs. Number of GPUs')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig('plots/ddp_speedup_comparison.png')
            plt.close()
            
            # Plot scaling efficiency
            plt.figure(figsize=(12, 6))
            
            for bs in ddp_data['Batch Size'].unique():
                bs_data = ddp_data[ddp_data['Batch Size'] == bs]
                if 1 in bs_data['GPUs'].values:
                    single_gpu_time = bs_data[bs_data['GPUs'] == bs_data['GPUs'].min()]['Training Time (Seconds)'].values[0]
                    bs_data = bs_data.copy()
                    bs_data['Efficiency'] = (single_gpu_time / bs_data['Training Time (Seconds)']) / bs_data['GPUs'] * 100
                    plt.plot(bs_data['GPUs'], bs_data['Efficiency'], marker='o', linewidth=2, label=f'Batch Size {bs}')
            
            plt.axhline(y=100, color='k', linestyle='--', label='Ideal Efficiency')
            plt.xlabel('Number of GPUs')
            plt.ylabel('Scaling Efficiency (%)')
            plt.title('DDP Scaling Efficiency vs. Number of GPUs')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig('plots/ddp_scaling_efficiency.png')
            plt.close()
            
            print("DDP comparison plots saved to 'plots/' directory")
        else:
            print("No DDP timing data found.")
    except Exception as e:
        print(f"Error generating DDP comparison plots: {e}")

def generate_cuda_kernel_analysis_plots():
    """Generate analysis plots from CUDA profiling data."""
    print("Generating CUDA kernel analysis plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Check if CUDA profiling data exists
    profiler_logs = Path('./profiler_logs')
    if not profiler_logs.exists():
        print("No profiler logs found. Run with --profile to generate profiling data.")
        return
    
    try:
        # Try to load profiling data for both models
        try:
            efficientnet_profiling = pd.read_csv('profiler_logs/efficientnet_b3_profiling_metrics.csv')
            has_efficientnet_data = True
        except:
            has_efficientnet_data = False
            print("No EfficientNetB3 profiling data found.")
            
        try:
            resnet_profiling = pd.read_csv('profiler_logs/resnet50_profiling_metrics.csv')
            has_resnet_data = True
        except:
            has_resnet_data = False
            print("No ResNet50 profiling data found.")
        
        if has_efficientnet_data:
            # Top 10 CUDA kernels by time for EfficientNetB3
            plt.figure(figsize=(14, 8))
            top_kernels = efficientnet_profiling.nlargest(10, 'cuda_time_total')
            sns.barplot(x='cuda_time_total', y='name', data=top_kernels)
            plt.title('Top 10 CUDA Kernels by Time - EfficientNetB3')
            plt.xlabel('CUDA Time (ms)')
            plt.ylabel('Kernel Name')
            plt.tight_layout()
            plt.savefig('plots/efficientnet_top_cuda_kernels.png')
            plt.close()
        
        if has_resnet_data:
            # Top 10 CUDA kernels by time for ResNet50
            plt.figure(figsize=(14, 8))
            top_kernels = resnet_profiling.nlargest(10, 'cuda_time_total')
            sns.barplot(x='cuda_time_total', y='name', data=top_kernels)
            plt.title('Top 10 CUDA Kernels by Time - ResNet50')
            plt.xlabel('CUDA Time (ms)')
            plt.ylabel('Kernel Name')
            plt.tight_layout()
            plt.savefig('plots/resnet_top_cuda_kernels.png')
            plt.close()
        
        if has_efficientnet_data and has_resnet_data:
            # Compare common kernels between both models
            common_kernels = pd.merge(
                efficientnet_profiling[['name', 'cuda_time_total']].rename(columns={'cuda_time_total': 'efficientnet_time'}),
                resnet_profiling[['name', 'cuda_time_total']].rename(columns={'cuda_time_total': 'resnet_time'}),
                on='name', how='inner'
            )
            
            if not common_kernels.empty:
                top_common = common_kernels.nlargest(10, ['efficientnet_time', 'resnet_time'])
                
                # Reshape for plotting
                melted = pd.melt(
                    top_common, 
                    id_vars=['name'], 
                    value_vars=['efficientnet_time', 'resnet_time'],
                    var_name='model', 
                    value_name='cuda_time'
                )
                
                plt.figure(figsize=(14, 8))
                sns.barplot(x='cuda_time', y='name', hue='model', data=melted)
                plt.title('Common CUDA Kernels - EfficientNetB3 vs ResNet50')
                plt.xlabel('CUDA Time (ms)')
                plt.ylabel('Kernel Name')
                plt.tight_layout()
                plt.savefig('plots/model_kernel_comparison.png')
                plt.close()
        
        # Generate kernel count analysis if data available
        if has_efficientnet_data or has_resnet_data:
            # Group kernels by type and count them
            kernel_types = ['conv', 'gemm', 'relu', 'pool', 'norm', 'softmax', 'add', 'other']
            
            if has_efficientnet_data:
                efficient_counts = {}
                for k_type in kernel_types:
                    count = efficientnet_profiling[efficientnet_profiling['name'].str.contains(k_type, case=False)].shape[0]
                    efficient_counts[k_type] = count
                
                # For 'other' category, subtract all counted types from total
                counted = sum(efficient_counts.values())
                total = efficientnet_profiling.shape[0]
                efficient_counts['other'] = total - counted + efficient_counts['other']
                
                # Plot kernel count distribution
                plt.figure(figsize=(12, 6))
                plt.bar(efficient_counts.keys(), efficient_counts.values())
                plt.title('CUDA Kernel Distribution by Type - EfficientNetB3')
                plt.xlabel('Kernel Type')
                plt.ylabel('Count')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.savefig('plots/efficientnet_kernel_distribution.png')
                plt.close()
            
            if has_resnet_data:
                resnet_counts = {}
                for k_type in kernel_types:
                    count = resnet_profiling[resnet_profiling['name'].str.contains(k_type, case=False)].shape[0]
                    resnet_counts[k_type] = count
                
                # For 'other' category, subtract all counted types from total
                counted = sum(resnet_counts.values())
                total = resnet_profiling.shape[0]
                resnet_counts['other'] = total - counted + resnet_counts['other']
                
                # Plot kernel count distribution
                plt.figure(figsize=(12, 6))
                plt.bar(resnet_counts.keys(), resnet_counts.values())
                plt.title('CUDA Kernel Distribution by Type - ResNet50')
                plt.xlabel('Kernel Type')
                plt.ylabel('Count')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.savefig('plots/resnet_kernel_distribution.png')
                plt.close()
            
            # If both models have data, compare their kernel distributions
            if has_efficientnet_data and has_resnet_data:
                plt.figure(figsize=(14, 7))
                x = np.arange(len(kernel_types))
                width = 0.35
                
                efficient_values = [efficient_counts[k] for k in kernel_types]
                resnet_values = [resnet_counts[k] for k in kernel_types]
                
                plt.bar(x - width/2, efficient_values, width, label='EfficientNetB3')
                plt.bar(x + width/2, resnet_values, width, label='ResNet50')
                
                plt.xlabel('Kernel Type')
                plt.ylabel('Count')
                plt.title('CUDA Kernel Distribution Comparison')
                plt.xticks(x, kernel_types)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.savefig('plots/kernel_distribution_comparison.png')
                plt.close()
                
        print("CUDA kernel analysis plots generated successfully.")
    except Exception as e:
        print(f"Error generating CUDA kernel analysis plots: {e}")

def generate_memory_usage_plots():
    """Generate plots analyzing memory usage and patterns."""
    print("Generating memory usage plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Sample memory data - in a real scenario, this would come from actual profiling
        batch_sizes = [16, 32, 64, 128, 256]
        
        # EfficientNetB3 memory by batch size (in MB)
        memory_efficientnet = {
            'weights': [120] * 5,  # Constant across batch sizes
            'activations': [240, 480, 960, 1920, 3840],  # Scales linearly with batch size
            'gradients': [200, 400, 800, 1600, 3200],  # Scales linearly with batch size
            'optimizer': [50] * 5,  # Constant across batch sizes
            'workspace': [100, 120, 150, 180, 220]  # Increases with batch size but not linearly
        }
        
        # ResNet50 memory by batch size (in MB)
        memory_resnet = {
            'weights': [100] * 5,  # Constant across batch sizes
            'activations': [200, 400, 800, 1600, 3200],  # Scales linearly with batch size
            'gradients': [180, 360, 720, 1440, 2880],  # Scales linearly with batch size
            'optimizer': [40] * 5,  # Constant across batch sizes
            'workspace': [80, 100, 130, 160, 200]  # Increases with batch size but not linearly
        }
        
        # 1. Total memory usage by batch size for both models
        plt.figure(figsize=(12, 6))
        
        # Calculate total memory for each model and batch size
        total_memory_efficientnet = [sum(values) for values in zip(
            memory_efficientnet['weights'],
            memory_efficientnet['activations'],
            memory_efficientnet['gradients'],
            memory_efficientnet['optimizer'],
            memory_efficientnet['workspace']
        )]
        
        total_memory_resnet = [sum(values) for values in zip(
            memory_resnet['weights'],
            memory_resnet['activations'],
            memory_resnet['gradients'],
            memory_resnet['optimizer'],
            memory_resnet['workspace']
        )]
        
        plt.plot(batch_sizes, total_memory_efficientnet, 'bo-', linewidth=2, label='EfficientNetB3')
        plt.plot(batch_sizes, total_memory_resnet, 'ro-', linewidth=2, label='ResNet50')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('GPU Memory Usage vs Batch Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add a hypothetical GPU memory limit line (e.g., 16GB)
        plt.axhline(y=16000, color='k', linestyle='--', label='16GB GPU Memory')
        plt.text(batch_sizes[0], 16500, '16GB VRAM', va='bottom', ha='left')
        
        plt.tight_layout()
        plt.savefig('plots/memory_vs_batch_size.png')
        plt.close()
        
        # 2. Memory breakdown by component for each model (fixed batch size of 64)
        plt.figure(figsize=(14, 7))
        
        # Get the index of batch size 64
        bs_idx = batch_sizes.index(64)
        
        # Prepare data for stacked bar chart
        categories = ['Weights', 'Activations', 'Gradients', 'Optimizer', 'Workspace']
        
        efficient_values = [
            memory_efficientnet['weights'][bs_idx],
            memory_efficientnet['activations'][bs_idx],
            memory_efficientnet['gradients'][bs_idx],
            memory_efficientnet['optimizer'][bs_idx],
            memory_efficientnet['workspace'][bs_idx]
        ]
        
        resnet_values = [
            memory_resnet['weights'][bs_idx],
            memory_resnet['activations'][bs_idx],
            memory_resnet['gradients'][bs_idx],
            memory_resnet['optimizer'][bs_idx],
            memory_resnet['workspace'][bs_idx]
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.bar(x - width/2, efficient_values, width, label='EfficientNetB3')
        ax.bar(x + width/2, resnet_values, width, label='ResNet50')
        
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('GPU Memory Usage Breakdown by Component (Batch Size = 64)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/memory_breakdown_comparison.png')
        plt.close()
        
        # 3. Memory scaling factors for different components
        plt.figure(figsize=(14, 7))
        
        # Calculate scaling factors relative to batch size 16
        base_idx = 0  # Index for batch size 16
        
        # EfficientNetB3 scaling factors
        efficient_scaling = {
            'activations': [v / memory_efficientnet['activations'][base_idx] for v in memory_efficientnet['activations']],
            'gradients': [v / memory_efficientnet['gradients'][base_idx] for v in memory_efficientnet['gradients']]
        }
        
        # ResNet50 scaling factors
        resnet_scaling = {
            'activations': [v / memory_resnet['activations'][base_idx] for v in memory_resnet['activations']],
            'gradients': [v / memory_resnet['gradients'][base_idx] for v in memory_resnet['gradients']]
        }
        
        # Theoretical perfect scaling (linear with batch size)
        perfect_scaling = [bs / batch_sizes[base_idx] for bs in batch_sizes]
        
        plt.plot(batch_sizes, efficient_scaling['activations'], 'bo-', linewidth=2, label='EfficientNetB3 Activations')
        plt.plot(batch_sizes, efficient_scaling['gradients'], 'b--', linewidth=2, label='EfficientNetB3 Gradients')
        plt.plot(batch_sizes, resnet_scaling['activations'], 'ro-', linewidth=2, label='ResNet50 Activations')
        plt.plot(batch_sizes, resnet_scaling['gradients'], 'r--', linewidth=2, label='ResNet50 Gradients')
        plt.plot(batch_sizes, perfect_scaling, 'k-', linewidth=1, label='Perfect Linear Scaling')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Scaling Factor (relative to BS=16)')
        plt.title('Memory Scaling Factors vs Batch Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/memory_scaling_factors.png')
        plt.close()
        
        print("Memory usage plots generated successfully.")
    except Exception as e:
        print(f"Error generating memory usage plots: {e}")

def generate_model_comparison_plots():
    """Generate comparative plots between EfficientNetB3 and ResNet50."""
    print("Generating model comparison plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Try to load model comparison results
        try:
            results_df = pd.read_csv('results/model_comparison_results.csv')
            has_comparison_data = True
        except:
            has_comparison_data = False
            print("No model comparison data found. Using sample data for demonstration.")
            
            # Create sample data
            results_df = pd.DataFrame({
                'model': ['efficientnet_b3', 'efficientnet_b3', 'efficientnet_b3', 
                         'resnet50', 'resnet50', 'resnet50'],
                'batch_size': [32, 64, 128, 32, 64, 128],
                'mixed_precision': [False, False, False, False, False, False],
                'avg_epoch_time': [120, 110, 105, 100, 90, 85],
                'val_acc': [78.5, 80.2, 79.8, 77.6, 78.9, 78.2]
            })
        
        # 1. Training time comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='batch_size', y='avg_epoch_time', hue='model', data=results_df)
        plt.title('Training Time per Epoch by Model and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Time per Epoch (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/model_training_time_comparison.png')
        plt.close()
        
        # 2. Validation accuracy comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='batch_size', y='val_acc', hue='model', data=results_df)
        plt.title('Validation Accuracy by Model and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Validation Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/model_accuracy_comparison.png')
        plt.close()
        
        # 3. Training efficiency (accuracy per time)
        results_df['efficiency'] = results_df['val_acc'] / results_df['avg_epoch_time']
        plt.figure(figsize=(12, 6))
        sns.barplot(x='batch_size', y='efficiency', hue='model', data=results_df)
        plt.title('Training Efficiency by Model and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Efficiency (Accuracy % / Second)')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('plots/model_efficiency_comparison.png')
        plt.close()
        
        # 4. Check for mixed precision data
        if 'mixed_precision' in results_df.columns and True in results_df['mixed_precision'].values:
            # Filter for batch size 64 (or the most common batch size)
            if 64 in results_df['batch_size'].unique():
                mp_data = results_df[results_df['batch_size'] == 64]
            else:
                most_common_bs = results_df['batch_size'].value_counts().idxmax()
                mp_data = results_df[results_df['batch_size'] == most_common_bs]
            
            # Plot mixed precision impact on training time
            plt.figure(figsize=(12, 6))
            sns.barplot(x='model', y='avg_epoch_time', hue='mixed_precision', data=mp_data)
            plt.title('Impact of Mixed Precision on Training Time')
            plt.xlabel('Model')
            plt.ylabel('Average Time per Epoch (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()
            plt.savefig('plots/mixed_precision_impact.png')
            plt.close()
            
            # Plot mixed precision impact on validation accuracy
            plt.figure(figsize=(12, 6))
            sns.barplot(x='model', y='val_acc', hue='mixed_precision', data=mp_data)
            plt.title('Impact of Mixed Precision on Validation Accuracy')
            plt.xlabel('Model')
            plt.ylabel('Validation Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()
            plt.savefig('plots/mixed_precision_accuracy_impact.png')
            plt.close()
        
        print("Model comparison plots generated successfully.")
    except Exception as e:
        print(f"Error generating model comparison plots: {e}")

def generate_kernel_execution_plots():
    """Generate plots analyzing kernel execution patterns."""
    print("Generating kernel execution pattern plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Sample data for kernel execution patterns (in a real scenario, this would come from profiling)
        kernel_data = {
            'efficientnet_b3': {
                'kernel_types': ['Convolution', 'Matrix Multiply', 'Pointwise', 'Memory Copy', 'Normalization', 'Reduction', 'Other'],
                'execution_count': [340, 270, 420, 180, 150, 120, 90],
                'total_time_ms': [320, 240, 110, 90, 60, 45, 35],
                'avg_duration_us': [940, 890, 260, 500, 400, 375, 390]
            },
            'resnet50': {
                'kernel_types': ['Convolution', 'Matrix Multiply', 'Pointwise', 'Memory Copy', 'Normalization', 'Reduction', 'Other'],
                'execution_count': [380, 240, 390, 160, 170, 110, 80],
                'total_time_ms': [350, 220, 95, 80, 70, 40, 30],
                'avg_duration_us': [920, 920, 240, 500, 410, 360, 380]
            }
        }
        
        # 1. Kernel execution count comparison
        plt.figure(figsize=(14, 7))
        
        x = np.arange(len(kernel_data['efficientnet_b3']['kernel_types']))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(x - width/2, kernel_data['efficientnet_b3']['execution_count'], width, label='EfficientNetB3')
        ax.bar(x + width/2, kernel_data['resnet50']['execution_count'], width, label='ResNet50')
        
        ax.set_ylabel('Execution Count')
        ax.set_title('CUDA Kernel Execution Count by Type')
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_data['efficientnet_b3']['kernel_types'])
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/kernel_execution_count.png')
        plt.close()
        
        # 2. Total kernel time comparison
        plt.figure(figsize=(14, 7))
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(x - width/2, kernel_data['efficientnet_b3']['total_time_ms'], width, label='EfficientNetB3')
        ax.bar(x + width/2, kernel_data['resnet50']['total_time_ms'], width, label='ResNet50')
        
        ax.set_ylabel('Total Time (ms)')
        ax.set_title('CUDA Kernel Total Execution Time by Type')
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_data['efficientnet_b3']['kernel_types'])
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/kernel_total_time.png')
        plt.close()
        
        # 3. Average kernel duration comparison
        plt.figure(figsize=(14, 7))
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(x - width/2, kernel_data['efficientnet_b3']['avg_duration_us'], width, label='EfficientNetB3')
        ax.bar(x + width/2, kernel_data['resnet50']['avg_duration_us'], width, label='ResNet50')
        
        ax.set_ylabel('Average Duration (µs)')
        ax.set_title('CUDA Kernel Average Duration by Type')
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_data['efficientnet_b3']['kernel_types'])
        ax.legend()
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/kernel_avg_duration.png')
        plt.close()
        
        # 4. Time distribution by kernel type
        plt.figure(figsize=(12, 10))
        
        # Create pie charts for time distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # EfficientNetB3 pie chart
        ax1.pie(kernel_data['efficientnet_b3']['total_time_ms'], 
               labels=kernel_data['efficientnet_b3']['kernel_types'],
               autopct='%1.1f%%',
               startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax1.set_title('EfficientNetB3 Kernel Time Distribution')
        
        # ResNet50 pie chart
        ax2.pie(kernel_data['resnet50']['total_time_ms'], 
               labels=kernel_data['resnet50']['kernel_types'],
               autopct='%1.1f%%',
               startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax2.set_title('ResNet50 Kernel Time Distribution')
        
        plt.tight_layout()
        plt.savefig('plots/kernel_time_distribution.png')
        plt.close()
        
        print("Kernel execution pattern plots generated successfully.")
    except Exception as e:
        print(f"Error generating kernel execution pattern plots: {e}")

def generate_hardware_utilization_plots():
    """Generate plots showing GPU utilization metrics."""
    print("Generating hardware utilization plots...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Sample data for GPU utilization (in a real scenario, this would come from profiling)
        # Time points during training
        time_points = np.linspace(0, 100, 20)
        
        # GPU SM (Streaming Multiprocessor) utilization percentage
        sm_util_efficient = 85 + 10 * np.sin(time_points/10)
        sm_util_efficient = np.clip(sm_util_efficient, 0, 100)
        
        sm_util_resnet = 90 + 8 * np.sin(time_points/8)
        sm_util_resnet = np.clip(sm_util_resnet, 0, 100)
        
        # GPU memory bandwidth utilization percentage
        mem_bw_efficient = 75 + 15 * np.sin(time_points/12 + 1)
        mem_bw_efficient = np.clip(mem_bw_efficient, 0, 100)
        
        mem_bw_resnet = 80 + 12 * np.sin(time_points/9 + 0.5)
        mem_bw_resnet = np.clip(mem_bw_resnet, 0, 100)
        
        # 1. SM Utilization over time
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_points, sm_util_efficient, 'b-', linewidth=2, label='EfficientNetB3')
        plt.plot(time_points, sm_util_resnet, 'r-', linewidth=2, label='ResNet50')
        
        plt.xlabel('Training Progress (%)')
        plt.ylabel('SM Utilization (%)')
        plt.title('GPU Streaming Multiprocessor Utilization During Training')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/gpu_sm_utilization.png')
        plt.close()
        
        # 2. Memory bandwidth utilization over time
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_points, mem_bw_efficient, 'b-', linewidth=2, label='EfficientNetB3')
        plt.plot(time_points, mem_bw_resnet, 'r-', linewidth=2, label='ResNet50')
        
        plt.xlabel('Training Progress (%)')
        plt.ylabel('Memory Bandwidth Utilization (%)')
        plt.title('GPU Memory Bandwidth Utilization During Training')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/gpu_memory_bandwidth.png')
        plt.close()
        
        # 3. Compute vs Memory bound analysis
        plt.figure(figsize=(14, 7))
        
        # Sample data for compute vs memory boundedness
        operations = ['Conv 1x1', 'Conv 3x3', 'Conv 5x5', 'Linear', 'BatchNorm', 'ReLU', 'Pooling']
        
        # Arithmetic intensity (FLOPS/Byte) - higher means more compute bound
        arith_intensity_efficient = [4.2, 6.8, 8.5, 2.1, 0.5, 0.2, 0.3]
        arith_intensity_resnet = [4.0, 7.2, 9.0, 2.0, 0.5, 0.2, 0.3]
        
        # Roofline threshold (example value, would depend on actual hardware)
        roofline_threshold = 3.0
        
        x = np.arange(len(operations))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - width/2, arith_intensity_efficient, width, label='EfficientNetB3')
        bars2 = ax.bar(x + width/2, arith_intensity_resnet, width, label='ResNet50')
        
        # Add a horizontal line for the roofline threshold
        ax.axhline(y=roofline_threshold, color='k', linestyle='--', label='Compute/Memory Bound Threshold')
        
        # Color the bars based on whether they're compute or memory bound
        for bars in [bars1, bars2]:
            for i, bar in enumerate(bars):
                if bar.get_height() >= roofline_threshold:
                    bar.set_color('green')  # Compute bound
                else:
                    bar.set_color('orange')  # Memory bound
        
        ax.set_ylabel('Arithmetic Intensity (FLOPS/Byte)')
        ax.set_title('Compute vs Memory Bound Analysis by Operation Type')
        ax.set_xticks(x)
        ax.set_xticklabels(operations)
        ax.legend()
        
        # Add text annotations above each bar
        for i, bars in enumerate([bars1, bars2]):
            for bar in bars:
                height = bar.get_height()
                boundedness = "Compute Bound" if height >= roofline_threshold else "Memory Bound"
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        boundedness, ha='center', va='bottom', rotation=90, fontsize=8)
        
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/compute_memory_bound_analysis.png')
        plt.close()
        
        print("Hardware utilization plots generated successfully.")
    except Exception as e:
        print(f"Error generating hardware utilization plots: {e}")

def main():
    print("Generating plots for CUDA profiling and model comparison...")
    
    # Generate all plots
    generate_training_metrics_plots()
    generate_ddp_comparison_plots()
    generate_cuda_kernel_analysis_plots()
    generate_memory_usage_plots()
    generate_model_comparison_plots()
    generate_kernel_execution_plots()
    generate_hardware_utilization_plots()
    
    print("All plots have been generated and saved to the 'plots/' directory.")

if __name__ == "__main__":
    main()