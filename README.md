# Fashion MNIST ANN Parallel Programing

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Execution Guide](#execution-guide)
4. [Contributors](#contributors)

## Project Overview

This project focuses on building a simple Artificial Neural Network (ANN) using CUDA for GPU acceleration. The primary goal is to compare the performance of the following approaches:
- Execution on the host (CPU).
- Execution on the GPU without optimization.
- Execution on the GPU with optimization techniques.

The dataset used for this project is the **FashionMNIST** dataset, which contains grayscale images of clothing items. Performance metrics such as training time, inference time, and accuracy are compared across the aforementioned approaches.

## Folder Structure

```
├── .git/
├── .gitignore
├── CPU.cu
├── dataset/
│   ├── test/
│   │   ├── t10k-images-idx3-ubyte
│   │   └── t10k-labels-idx1-ubyte
│   └── train/
│       ├── train-images-idx3-ubyte
│       └── train-labels-idx1-ubyte
├── GPU.cu
├── GPUv1.cu
├── GPUv2.cu
├── Makefile
├── modules/
│   ├── nnCPU.cu
│   ├── nnGPU.cu
│   ├── nnO1.cu
│   └── nnO2.cu
├── README.md
├── Report.ipynb
├── script.txt
└── utils/
    ├── train.cu
    ├── utils.cu
    └── utils.h
```

## Execution Guide

### Prerequisites
1. **Hardware**: A machine with a compatible NVIDIA GPU is required for GPU implementations.
2. **Software**:
   - NVIDIA CUDA Toolkit.

### Steps to Run

#### 1. Setting Up the Environment
- Clone the repository:
```bash
git clone https://github.com/tcbao0/Fashion-MNIST-ANN-Parallel-Programing.git
cd Fashion-MNIST-ANN-Parallel-Programing
```

- Download dataset here and unzip it.
- Compile it:
```bash
make && cd run
```
   + Run with Default Epoch (10):
   Use the following command:
```bash
./run_cpu
```

   + Run with a Custom Epoch Count:
   You can specify the number of epochs as an argument:
```bash
./run_gpu {epochs}
```

## Contributors

- Trần Thuận Phát
- Trần Công Bảo
- Đinh Công Huy Hoàng
