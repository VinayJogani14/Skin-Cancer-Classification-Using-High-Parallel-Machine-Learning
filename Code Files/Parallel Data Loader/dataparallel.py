import os
import time
import torch
import torch.multiprocessing as mp
from multiprocessing import freeze_support
from PIL import Image
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # For progress bars
import gc  # For garbage collection

# ---------------------------
# Check GPU availability and properties
def check_gpus():
    if not torch.cuda.is_available():
        print("No CUDA GPUs available. This script requires CUDA GPUs.")
        return False
        
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} CUDA GPUs:")
    
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {gpu_props.name} with {gpu_props.total_memory/1e9:.2f} GB memory")
    
    return gpu_count

# ---------------------------
# Get Image Paths
def get_image_paths(dataset_path="/scratch/panchani.d/Hpc/dataset"):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths

# ---------------------------
# Define a GPU-aware transform for faster processing
def get_gpu_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

# ---------------------------
# Define a GPU-aware dataset
class GPUImageDataset(Dataset):
    def __init__(self, image_paths, device):
        self.paths = image_paths
        self.device = device
        self.transform = get_gpu_transform()
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            img_tensor = self.transform(img)
            
            # Don't move tensor to device here - will be done in the processing loop
            # to avoid CUDA initialization issues in worker processes
            return img_tensor
            
        except Exception:
            # Return a valid tensor on CPU
            return torch.zeros(3, 224, 224)

# ---------------------------
# Process a batch on GPU
def process_batch_on_gpu(batch, device):
    # Do some actual GPU processing (normalization + simple convolution)
    batch = batch * 2.0 - 1.0  # Normalize to [-1, 1]
    
    # Simple convolution kernel - one per channel
    kernel = torch.ones((3, 1, 3, 3), device=device) / 9.0  # 3x3 averaging kernel
    
    # Apply convolution to each channel separately and reshape correctly
    processed = []
    for c in range(3):  # Process each channel
        # Extract one channel and ensure it has the right dimensions
        channel = batch[:, c:c+1, :, :]  # Shape: [batch_size, 1, 224, 224]
        
        # Apply 2D convolution to this channel
        channel_processed = torch.nn.functional.conv2d(
            channel,
            kernel[c:c+1],  # Use just one kernel for this channel
            padding=1  # Keep the same dimensions
        )
        processed.append(channel_processed)
    
    # Concatenate the processed channels
    batch_processed = torch.cat(processed, dim=1)
    
    # Return a small result to avoid transferring large data back to CPU
    return batch_processed.mean().item()

# ---------------------------
# Function to run a single GPU test with PyTorch DataLoader
def run_single_gpu_dataloader(image_paths, gpu_id, worker_count):
    """Run DataLoader on a single specified GPU"""
    device = torch.device(f'cuda:{gpu_id}')
    
    try:
        # Create dataset without attaching it to device
        dataset = GPUImageDataset(image_paths, device)
        loader = DataLoader(
            dataset, 
            batch_size=32, 
            num_workers=worker_count,
            pin_memory=True,  # Use pin memory for faster transfers
            persistent_workers=True
        )
        
        start = time.time()
        
        # Process all batches - move to device here, not in the dataset
        for batch in tqdm(loader, desc=f"DataLoader (GPU {gpu_id}, {worker_count} workers)"):
            batch = batch.to(device)  # Move to GPU here, not in __getitem__
            process_batch_on_gpu(batch, device)
            
        duration = time.time() - start
        
        # Clean up
        del dataset, loader
        torch.cuda.empty_cache()
        gc.collect()
        
        return duration
        
    except Exception as e:
        print(f"DataLoader on GPU {gpu_id} failed: {e}")
        return float('inf')

# ---------------------------
# Joblib with GPU acceleration
def run_joblib_gpu(image_paths, worker_count, gpu_id=0):
    """Run joblib with a single GPU for acceleration"""
    print(f"\nStarting Joblib processing with {worker_count} workers on GPU {gpu_id}...")
    
    # Set the CUDA device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Pre-define the transform
    transform = get_gpu_transform()
    
    def process_with_gpu(path):
        try:
            # Load and transform image
            img = Image.open(path).convert('RGB')
            tensor = transform(img)
            
            # Move to GPU inside this function (after subprocess is created)
            tensor = tensor.to(device)
            
            # Do some GPU processing
            tensor = tensor * 2.0 - 1.0
            
            # Simple convolution - one per channel
            kernel = torch.ones((3, 1, 3, 3), device=device) / 9.0
            processed = []
            
            for c in range(3):  # Process each channel
                # Extract one channel
                channel = tensor[c:c+1].unsqueeze(0)  # Shape: [1, 1, 224, 224]
                
                # Apply 2D convolution to this channel
                channel_processed = torch.nn.functional.conv2d(
                    channel,
                    kernel[c:c+1],
                    padding=1
                )
                processed.append(channel_processed)
            
            # Concatenate the processed channels
            tensor_processed = torch.cat(processed, dim=1)
            
            # Return a small result to avoid large data transfer
            return tensor_processed.mean().item()
            
        except Exception as e:
            print(f"Error in process_with_gpu: {e}")
            return 0.0
    
    start = time.time()
    try:
        with tqdm(total=len(image_paths), desc=f"Joblib (GPU {gpu_id}, {worker_count} workers)") as pbar:
            # Define a wrapper function that updates the progress bar
            def process_with_progress(path):
                result = process_with_gpu(path)
                pbar.update(1)
                return result
            
            # Use threads for Joblib when working with CUDA
            results = Parallel(
                n_jobs=worker_count,      # CPU worker count
                batch_size=32,            # Batch size
                timeout=None,             # No timeout
                prefer="threads",         # Use threads for GPU access - THIS IS CRITICAL
                verbose=0                 # Reduce verbosity
            )(delayed(process_with_progress)(p) for p in image_paths)
        
        duration = time.time() - start
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return duration
        
    except Exception as e:
        print(f"Joblib GPU failed: {e}")
        return float('inf')

# ---------------------------
# Simple loop with GPU acceleration (baseline)
def run_simple_loop_gpu(image_paths, gpu_id=0):
    """Run a simple loop with GPU acceleration for baseline comparison"""
    print(f"\nStarting Simple Loop with GPU {gpu_id} (baseline)...")
    
    # Set the CUDA device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Pre-define the transform
    transform = get_gpu_transform()
    
    try:
        start = time.time()
        
        for path in tqdm(image_paths, desc=f"Simple Loop GPU {gpu_id}"):
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img)
                
                # Move to device here (after opening the image)
                tensor = tensor.to(device)
                
                # Do some GPU processing
                tensor = tensor * 2.0 - 1.0
                
                # Simple convolution - one per channel
                kernel = torch.ones((3, 1, 3, 3), device=device) / 9.0
                processed = []
                
                for c in range(3):  # Process each channel
                    # Extract one channel
                    channel = tensor[c:c+1].unsqueeze(0)  # Shape: [1, 1, 224, 224]
                    
                    # Apply 2D convolution to this channel
                    channel_processed = torch.nn.functional.conv2d(
                        channel,
                        kernel[c:c+1],
                        padding=1
                    )
                    processed.append(channel_processed)
                
                # Concatenate the processed channels
                tensor_processed = torch.cat(processed, dim=1)
                
                # Get result (avoid accumulating tensors)
                result = tensor_processed.mean().item()
                
                # Explicitly delete tensors to prevent memory accumulation
                del tensor, tensor_processed, processed, channel_processed
                
            except Exception as e:
                print(f"Error processing image: {e}")
                pass
                
        duration = time.time() - start
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return duration
        
    except Exception as e:
        print(f"Simple Loop GPU failed: {e}")
        return float('inf')

# ---------------------------
# Multi-GPU parallel processing with DataLoader
# This function is moved to run inside a subprocess
def run_multi_gpu_dataloader_worker(gpu_id, image_paths, worker_count, result_queue):
    """Worker function to run a DataLoader on a specific GPU"""
    try:
        duration = run_single_gpu_dataloader(image_paths, gpu_id, worker_count)
        result_queue.put((gpu_id, duration))
    except Exception as e:
        print(f"Error in GPU {gpu_id} worker: {e}")
        result_queue.put((gpu_id, float('inf')))

def run_multi_gpu_dataloader(image_paths, num_gpus, worker_count=2):
    """Run processing across multiple GPUs using separate processes"""
    if num_gpus <= 0:
        return float('inf')
    
    # Split images across GPUs
    split_size = len(image_paths) // num_gpus
    splits = []
    
    for i in range(num_gpus):
        start_idx = i * split_size
        end_idx = (i+1) * split_size if i < num_gpus - 1 else len(image_paths)
        splits.append(image_paths[start_idx:end_idx])
    
    # Create a separate process for each GPU to avoid CUDA conflicts
    start = time.time()
    
    # Use multiprocessing.Process directly with a Queue
    from multiprocessing import Process, Queue
    result_queue = Queue()
    processes = []
    
    print(f"Starting Multi-GPU DataLoader with {num_gpus} GPUs, {worker_count} workers/GPU...")
    
    for i in range(num_gpus):
        p = Process(target=run_multi_gpu_dataloader_worker, 
                   args=(i, splits[i], worker_count, result_queue))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Calculate total duration - it's the max of all GPU durations
    durations = [duration for _, duration in results]
    if not durations or all(d == float('inf') for d in durations):
        print(f"All GPU workers failed")
        return float('inf')
    
    duration = max(d for d in durations if d != float('inf'))
    end = time.time()
    
    print(f"Multi-GPU DataLoader with {num_gpus} GPUs completed in {duration:.2f} seconds")
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return duration

# ---------------------------
# Function for multi-GPU with manual data splitting (alternative approach)
def run_multi_gpu_manual_worker(gpu_id, image_paths, worker_count, result_queue):
    """Worker function for a single GPU in manual processing approach"""
    try:
        duration = run_single_gpu_dataloader(image_paths, gpu_id, worker_count)
        result_queue.put((gpu_id, duration))
    except Exception as e:
        print(f"Error in GPU {gpu_id} manual worker: {e}")
        result_queue.put((gpu_id, float('inf')))

def run_multi_gpu_manual(image_paths, num_gpus, worker_count=2):
    """Manually split data across GPUs and process in parallel"""
    if num_gpus <= 0:
        return float('inf')
    
    try:
        # Split images across GPUs
        split_size = len(image_paths) // num_gpus
        splits = []
        
        for i in range(num_gpus):
            start_idx = i * split_size
            end_idx = (i+1) * split_size if i < num_gpus - 1 else len(image_paths)
            splits.append(image_paths[start_idx:end_idx])
        
        # Use the same approach as multi-GPU DataLoader
        from multiprocessing import Process, Queue
        result_queue = Queue()
        processes = []
        
        print(f"Starting Multi-GPU Manual with {num_gpus} GPUs, {worker_count} workers/GPU...")
        
        start = time.time()
        
        for i in range(num_gpus):
            p = Process(target=run_multi_gpu_manual_worker, 
                       args=(i, splits[i], worker_count, result_queue))
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # The duration is determined by the slowest GPU
        durations = [duration for _, duration in results]
        if not durations or all(d == float('inf') for d in durations):
            print(f"All GPU workers failed")
            return float('inf')
        
        duration = max(d for d in durations if d != float('inf'))
        
        print(f"Multi-GPU Manual with {num_gpus} GPUs completed in {duration:.2f} seconds")
        
        return duration
        
    except Exception as e:
        print(f"Manual multi-GPU failed: {e}")
        return float('inf')

# ---------------------------
# Main function to benchmark GPU performance
def main():
    # Set multiprocessing start method to 'spawn'
    # This must be set at the beginning before any other multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Check GPU availability and properties
    GPU_COUNT = check_gpus()
    if not GPU_COUNT:
        print("Exiting due to lack of CUDA GPUs")
        exit(1)

    # Ensure we test with at most 4 GPUs (as mentioned in the requirements)
    MAX_GPU_COUNT = min(4, GPU_COUNT)
    print(f"Will test with 1 to {MAX_GPU_COUNT} GPUs")

    # Get image paths
    print("Getting image paths...")
    DATASET_PATH = "/scratch/panchani.d/Hpc/dataset"
    image_paths = get_image_paths(DATASET_PATH)
    total_images = len(image_paths)
    print(f"Found {total_images} total images")

    # Limit to 50,000 images
    if len(image_paths) > 50000:
        image_paths = image_paths[:50000]
        
    print(f"Using exactly 50,000 images for benchmarking" if len(image_paths) >= 50000 else 
          f"Using all {len(image_paths)} images (less than 50,000 available)")

    # ---------------------------
    # Run all benchmarks
    results = []

    # 1. Single GPU tests with different methods and worker counts

    # 1.1 Simple loop on single GPU (baseline)
    gpu_baseline_duration = run_simple_loop_gpu(image_paths, 0)
    results.append((f"Simple Loop (GPU 0)", gpu_baseline_duration))

    # 1.2 Joblib on single GPU with varying worker counts
    for worker_count in [2, 4, 8]:
        duration = run_joblib_gpu(image_paths, worker_count, 0)
        results.append((f"Joblib (1 GPU, {worker_count} workers)", duration))

    # 1.3 DataLoader on single GPU with varying worker counts  
    for worker_count in [2, 4, 8]:
        duration = run_single_gpu_dataloader(image_paths, 0, worker_count)
        results.append((f"DataLoader (1 GPU, {worker_count} workers)", duration))

    # 2. Multi-GPU tests with different worker counts

    # 2.1 Multi-GPU DataLoader (1, 2, 3, 4 GPUs)
    for num_gpus in range(1, MAX_GPU_COUNT + 1):
        for worker_count in [2, 4]:
            duration = run_multi_gpu_dataloader(image_paths, num_gpus, worker_count)
            results.append((f"Multi-GPU DataLoader ({num_gpus} GPUs, {worker_count} workers/GPU)", duration))

    # 2.2 Multi-GPU Manual Split (1, 2, 3, 4 GPUs)
    for num_gpus in range(1, MAX_GPU_COUNT + 1):
        for worker_count in [2, 4]:
            duration = run_multi_gpu_manual(image_paths, num_gpus, worker_count)
            results.append((f"Multi-GPU Manual ({num_gpus} GPUs, {worker_count} workers/GPU)", duration))

    # ---------------------------
    # Process results and generate plots
    valid_results = [(technique, time) for technique, time in results if time != float('inf')]
    
    # Create plots if we have valid results
    if valid_results:
        df = pd.DataFrame(valid_results, columns=["Technique", "TimeTaken"])
        
        if len(df) > 0:  # Check if we have at least one valid result
            # Calculate speedups relative to single GPU baseline if it exists
            if 'Simple Loop (GPU 0)' in df['Technique'].values:
                gpu_baseline = df[df['Technique'] == 'Simple Loop (GPU 0)']['TimeTaken'].values[0]
                df['SpeedupVsGPU'] = gpu_baseline / df['TimeTaken']
            else:
                # If no baseline, just use relative comparison
                max_time = df['TimeTaken'].max()
                df['SpeedupVsGPU'] = max_time / df['TimeTaken']
                
            df.to_csv("gpu_parallel_benchmark_results.csv", index=False)
            
            # Add method and GPU count columns for better analysis
            df['Method'] = df['Technique'].apply(lambda x: x.split('(')[0].strip() if '(' in x else x)
            
            def extract_gpu_count(technique):
                if 'GPU 0' in technique:
                    return 1
                if 'GPUs' in technique:
                    # Extract the number before "GPUs"
                    parts = technique.split('(')[1].split(' ')
                    for part in parts:
                        if part.isdigit():
                            return int(part)
                return 1  # Default for single GPU
            
            df['GPUCount'] = df['Technique'].apply(extract_gpu_count)
            
            # ---------------------------
            # Create plots
            
            # 1. Method comparison (fastest configuration of each)
            plt.figure(figsize=(14, 7))
            method_groups = df.groupby('Method')['TimeTaken'].min().reset_index()
            sorted_methods = method_groups.sort_values('TimeTaken')
            
            plt.bar(sorted_methods["Method"], sorted_methods["TimeTaken"])
            plt.title(f"Time Taken by Each Method Type - Best Configuration (50K images)")
            plt.ylabel("Time (s)")
            plt.xlabel("Method")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("gpu_method_time_comparison.png")
            
            # 2. GPU Count vs Performance for multi-GPU methods
            plt.figure(figsize=(14, 7))
            
            # Filter for multi-GPU methods
            multi_gpu_df = df[df['Method'].str.contains('Multi-GPU')]
            
            # Group by method and GPU count - only if we have multi-GPU methods
            if not multi_gpu_df.empty:
                for method in multi_gpu_df['Method'].unique():
                    method_data = multi_gpu_df[multi_gpu_df['Method'] == method]
                    best_by_gpu = method_data.groupby('GPUCount')['TimeTaken'].min().reset_index()
                    plt.plot(best_by_gpu['GPUCount'], best_by_gpu['TimeTaken'], marker='o', label=method)
                
                # Add legend only if we have data
                if len(plt.gca().get_lines()) > 0:
                    plt.legend()
            
            plt.title(f"Processing Time vs GPU Count (50K images)")
            plt.xlabel("Number of GPUs")
            plt.ylabel("Time (s)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("gpu_scaling.png")
            
            # 3. Speedup chart relative to single GPU baseline
            plt.figure(figsize=(14, 7))
            
            # Sort by speedup
            speedup_df = df.sort_values('SpeedupVsGPU', ascending=False)
            
            # Filter out the baseline itself
            speedup_df = speedup_df[speedup_df['Technique'] != 'Simple Loop (GPU 0)']
            
            plt.bar(speedup_df["Technique"][:10], speedup_df["SpeedupVsGPU"][:10])  # Show top 10 for clarity
            plt.title(f"Speedup vs Single GPU Baseline (50K images)")
            plt.ylabel("Speedup Factor")
            plt.xlabel("Technique")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("gpu_speedup_comparison.png")
            
            # 4. Plot techniques by worker count
            plt.figure(figsize=(14, 7))
            
            # Get worker count from the technique name - safer parsing
            df['WorkerCount'] = df['Technique'].apply(
                lambda x: int(x.split('workers')[0].strip().split(' ')[-1]) 
                if 'workers' in x and x.split('workers')[0].strip().split(' ')[-1].isdigit() 
                else 0
            )
            
            # Filter for methods with worker variation
            worker_methods = df[df['WorkerCount'] > 0]
            
            for method in worker_methods['Method'].unique():
                method_data = worker_methods[worker_methods['Method'] == method]
                if 'Multi-GPU' in method:
                    # For multi-GPU methods, group by GPU count and worker count
                    for gpu_count in method_data['GPUCount'].unique():
                        gpu_data = method_data[method_data['GPUCount'] == gpu_count]
                        if len(gpu_data) > 1:  # Only plot if we have multiple worker counts
                            plt.plot(
                                gpu_data['WorkerCount'], 
                                gpu_data['TimeTaken'], 
                                marker='o', 
                                label=f"{method} ({gpu_count} GPUs)"
                            )
                else:
                    # For single GPU methods
                    plt.plot(
                        method_data['WorkerCount'], 
                        method_data['TimeTaken'], 
                        marker='o', 
                        label=method
                    )
            
            plt.title(f"Processing Time vs Worker Count (50K images)")
            plt.xlabel("Number of Workers")
            plt.ylabel("Time (s)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("gpu_worker_scaling.png")
            
            # 5. Create a plot specifically for GPU scaling efficiency
            if MAX_GPU_COUNT > 1:
                plt.figure(figsize=(14, 7))
                
                # Calculate ideal scaling line
                max_gpus = range(1, MAX_GPU_COUNT + 1)
                
                # Filter for methods with multiple GPU counts
                for method in multi_gpu_df['Method'].unique():
                    method_data = multi_gpu_df[multi_gpu_df['Method'] == method]
                    if len(method_data['GPUCount'].unique()) > 1:
                        # Get the single GPU time as baseline for this method
                        single_gpu_time = method_data[method_data['GPUCount'] == 1]['TimeTaken'].min()
                        
                        # Calculate actual times and theoretical ideal times
                        gpu_counts = []
                        actual_times = []
                        ideal_times = []
                        
                        for gpu_count in sorted(method_data['GPUCount'].unique()):
                            gpu_time = method_data[method_data['GPUCount'] == gpu_count]['TimeTaken'].min()
                            gpu_counts.append(gpu_count)
                            actual_times.append(gpu_time)
                            ideal_times.append(single_gpu_time / gpu_count)
                        
                        # Plot actual times
                        plt.plot(gpu_counts, actual_times, marker='o', label=f"{method} (Actual)")
                        
                        # Plot ideal scaling
                        plt.plot(gpu_counts, ideal_times, linestyle='--', 
                                 label=f"{method} (Ideal)", alpha=0.6)
                
                plt.title(f"GPU Scaling Efficiency (50K images)")
                plt.xlabel("Number of GPUs")
                plt.ylabel("Processing Time (s)")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig("gpu_scaling_efficiency.png")
            
            # ---------------------------
            # Print results table
            
            print("\nResults Summary (Sorted by Speed):")
            print("-" * 80)
            print(f"{'Technique':<48} {'Time (s)':<12} {'Speedup vs 1 GPU':<18}")
            print("-" * 80)
            
            for _, row in df.sort_values("TimeTaken").iterrows():
                technique = row['Technique']
                time_taken = row['TimeTaken']
                speedup_vs_gpu = row['SpeedupVsGPU']
                
                print(f"{technique:<48} {time_taken:<12.2f} {speedup_vs_gpu:<18.2f}")
            
            # Identify the fastest method
            fastest = df.loc[df["TimeTaken"].idxmin()]
            print("\nFastest method:", fastest["Technique"])
            print(f"Time: {fastest['TimeTaken']:.2f} seconds")
            print(f"Speedup vs. single GPU baseline: {fastest['SpeedupVsGPU']:.2f}x")
            
            # Compare scaling efficiency
            if MAX_GPU_COUNT > 1:
                multi_gpu_methods = df[df['Method'].str.contains('Multi-GPU')]
                
                print("\nGPU Scaling Efficiency:")
                print("-" * 80)
                print(f"{'Method':<30} {'1→{0} GPUs':<15} {'Efficiency':<10} {'Ideal':<10}".format(MAX_GPU_COUNT))
                print("-" * 80)
                
                for method in multi_gpu_methods['Method'].unique():
                    method_data = multi_gpu_methods[multi_gpu_methods['Method'] == method]
                    
                    if 1 in method_data['GPUCount'].values and MAX_GPU_COUNT in method_data['GPUCount'].values:
                        one_gpu = method_data[method_data['GPUCount'] == 1]['TimeTaken'].min()
                        max_gpu = method_data[method_data['GPUCount'] == MAX_GPU_COUNT]['TimeTaken'].min()
                        
                        speedup = one_gpu / max_gpu
                        scaling_efficiency = speedup / MAX_GPU_COUNT
                        
                        print(f"{method:<30} {speedup:<15.2f}x {scaling_efficiency:<10.2f} {1.0:<10.2f}")
                
    else:
        print("No valid results to plot")

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    # This is critical for multiprocessing to work properly in Windows
    freeze_support()
    main()