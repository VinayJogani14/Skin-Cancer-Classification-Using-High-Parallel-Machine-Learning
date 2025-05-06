#!/bin/bash

# Directory to save logs
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# GPU configurations to test
GPU_CONFIGS=(1 2 4)
BATCH_SIZES=(32 64 128)

# Number of epochs (can be adjusted)
EPOCHS=10

# Loop through all configurations
for gpus in "${GPU_CONFIGS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "Running experiment with $gpus GPUs and batch size $bs"
        
        # Create a unique log file name
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="$LOG_DIR/gpu${gpus}_bs${bs}_${TIMESTAMP}.log"
        
        # Run the training command
        torchrun --nproc_per_node=$gpus DDP.py \
            --batch_size $bs | tee $LOG_FILE
        
        echo "Experiment completed. Log saved to $LOG_FILE"
        echo "--------------------------------------------------"
    done
done

echo "All experiments completed!"