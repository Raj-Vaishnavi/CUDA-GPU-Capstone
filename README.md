# CUDA-Accelerated Batch Image Processing and GPU Performance Analysis

## Project Overview

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA. The application processes RGB images using CUDA kernels and performs multiple image-processing operations in parallel.

The project was developed to explore how GPU parallelism can be applied to image processing and how execution performance changes with different image sizes and operations.

The application supports:

- Grayscale conversion
- Brightness adjustment
- Blur filtering
- GPU execution-time measurement
- Command-line arguments
- Processing images of different sizes

## GPU Environment

The CUDA application was executed using an NVIDIA Tesla T4 GPU in Google Colab.

Example GPU configuration:

- GPU: NVIDIA Tesla T4
- Compute Capability: 7.5
- CUDA Toolkit: 12.8
- GPU Memory: 15 GB

The project was developed using Visual Studio Code and executed in a CUDA-enabled Google Colab environment.

## Technologies Used

- C++
- NVIDIA CUDA
- CUDA Runtime API
- `nvcc`
- Google Colab
- GitHub

## Project Structure

```text
CUDA-GPU-Capstone/
│
├── src/
│   ├── main.cu
│   ├── image_processing.cu
│   └── image_processing.cuh
│
├── data/
│   └── input/
│
├── results/
│   ├── performance.csv
│   ├── execution_log.txt
│   └── processed images
│
├── README.md
├── Makefile
└── run.sh
