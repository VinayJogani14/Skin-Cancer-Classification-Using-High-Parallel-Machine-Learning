# 🧠 Parallel Deep Learning for Dermatological Disease Classification

## Overview

This project explores the application of **distributed deep learning** for the classification of **35 distinct dermatological diseases** using **clinical image data**. By leveraging **PyTorch Distributed Data Parallel (DDP)** and **Automatic Mixed Precision (AMP)**, we accelerate training on a large-scale dataset (~245,000 images) using **multi-GPU computing**. The goal is to deliver fast, accurate, and scalable classification performance suitable for real-world medical AI deployment.

Developed for **CSYE 7105 – High-Performance Parallel Machine Learning and AI (Spring 2025)** at **Northeastern University**, under the supervision of **Prof. Handan Liu**.

---

## 🚀 Project Highlights

- **✅ High Accuracy**: Achieved **91.8% classification accuracy** and **macro F1 score of 96.2%** across 35 disease categories.
- **⚡ Scalable Training**: Achieved **3.3× speedup using 4 NVIDIA A100 GPUs** with ~84% parallel efficiency.
- **🧠 Advanced Architectures**: Used **EfficientNet-B3** and **DenseNet-121** with **ImageNet pretraining**.
- **🧪 Robust Evaluation**: Included per-class metrics, ROC curves, confusion matrices, and misclassification analysis.
- **💾 Optimized Data Pipeline**: Multi-threaded `DataLoader` with `DistributedSampler` for seamless multi-GPU data handling.
- **📉 Mixed Precision Training**: Reduced memory usage by 50% without compromising model accuracy.

---

## 📂 Dataset Information

- **Source**: Kaggle – [Multiple Skin Disease Detection and Classification](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification)
- **Size**: ~245,000 color clinical images
- **Classes**: 35 disease categories including:
  - **Benign and Malignant tumors** (e.g., Melanoma, BCC)
  - **Infections** (Fungal, Bacterial, Viral)
  - **Inflammatory diseases** (Psoriasis, Eczema)
  - **Others** (Hair loss, Pigmentation disorders, Drug eruptions)

### Directory Format

```
/dataset
├── acne/
├── melanoma/
├── eczema/
├── chickenpox/
├── ... [35 folders]
```

Each folder contains hundreds to thousands of JPEG images. Total: ~11GB.

---

## 🧪 Preprocessing Pipeline

| Step               | Details                                                  |
|--------------------|----------------------------------------------------------|
| Resize             | Images resized to 256×256                                |
| Crop               | RandomResizedCrop(224) (train), CenterCrop(224) (eval)  |
| Augmentation       | RandomHorizontalFlip (train only)                        |
| Normalization      | ImageNet mean & std (RGB)                                |
| Loader             | PyTorch `DataLoader` + `DistributedSampler`              |

---

## 🏗️ Model Architecture

We used two CNN architectures:

### 🔹 DenseNet-121
- Feature reuse via dense connectivity
- Excellent baseline for subtle visual feature classification

### 🔹 EfficientNet-B3 (Primary Model)
- Compound scaling of width, depth, resolution
- Strong accuracy with fewer parameters
- **Modified output layer for 35-class classification**
- Pretrained on ImageNet

---

## 🛠️ Training Configuration

| Parameter         | Value                           |
|-------------------|---------------------------------|
| Framework         | PyTorch                         |
| GPUs              | 4 × NVIDIA A100                 |
| Parallelism       | DistributedDataParallel (DDP)   |
| Mixed Precision   | AMP (Autocast + GradScaler)     |
| Optimizer         | Adam (`lr=0.001`)               |
| Loss Function     | CrossEntropyLoss                |
| Batch Size        | 64 per GPU (global = 256)       |
| Epochs            | 5 (early stopping at epoch 4)   |
| Scheduler         | None (manual tuning)            |

---

## 🔁 Reproducibility

- Fixed random seed on all processes
- Deterministic convolution settings via `torch.backends.cudnn`
- Synced training with `dist.barrier()` and `DistributedSampler.set_epoch()`
- All metrics, models, and training logs versioned and saved

---

## ⏱️ Training Performance

### ⌛ Training Time

| GPUs | Batch / GPU | Total Time (5 Epochs) | Speedup | Efficiency |
|------|-------------|------------------------|---------|------------|
| 1    | 32          | 3548.2s                | 1×      | 100%       |
| 2    | 32          | 1562.4s                | 2.27×   | 113%       |
| 4    | 32          | 1051.7s                | 3.37×   | 84.3%      |

---

## 🎯 Results

### 🧮 Accuracy Metrics

- **Overall Accuracy**: 91.8%
- **Macro F1 Score**: 96.2%
- **Micro ROC-AUC**: ~0.999

### 🧪 Per-Class Performance (sample)

| Class                    | Precision | Recall | F1     |
|--------------------------|-----------|--------|--------|
| Chickenpox               | 100%      | 99.99% | 100%   |
| Benign Lesions           | 93.65%    | 91.46% | 92.54% |
| Malignant Lesions        | 94.84%    | 93.70% | 94.27% |
| Warts/Molluscum (Lowest) | 81.88%    | 71.97% | 76.61% |

---

## ❌ Misclassification Analysis

| True Class     | Predicted As     | Count |
|----------------|------------------|-------|
| Benign         | Malignant        | 108   |
| Malignant      | Benign           | 80    |
| Psoriasis      | Eczema           | 40    |
| Nail Fungus    | Hair Disorders   | 101   |

---

## 💻 Installation

```bash
git clone https://github.com/VinayJogani14/parallel-dermatology-cnn.git
cd parallel-dermatology-cnn
pip install -r requirements.txt
```

---

## 🧪 Run Training

### 🔹 Multi-GPU Training (Recommended)

```bash
torchrun --nproc_per_node=4 train.py --config configs/effnet_b3_ddp.yaml
```

### 🔹 Single-GPU Baseline

```bash
python train.py --config configs/effnet_b3_single.yaml
```

---

## 📤 Inference

```python
from model import load_model, predict
image = "sample_image.jpg"
label, prob = predict(image, model_path="models/efficientnet_b3.pt")
```

> Inference speed: **600+ images/sec** on A100 GPU using AMP

---

## 🧑‍💻 Authors

- **Vinay Jogani** – [GitHub](https://github.com/vinayjogani)
- **Dhwanil Panchani**

Instructor: **Prof. Handan Liu**, Northeastern University

---


## 🔮 Future Work

- Scale to **multi-node training** for larger datasets
- Deploy as **REST API or mobile app**
- Apply **Grad-CAM** for model interpretability
- Use **ensemble learning** to reduce misclassification in low-performing classes
- Add **metadata inputs** (e.g., age, gender, location) for context-aware classification

---
