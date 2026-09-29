# CUDA-Accelerated Image Processing and GPU Performance Analysis

## Project Overview

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA.

The application processes RGB images using CUDA kernels and performs three image-processing operations:

- Grayscale conversion
- Brightness adjustment
- 3×3 blur filtering

Each image-processing operation is executed in parallel on the GPU, with one CUDA thread assigned to each image pixel.

The project also measures GPU execution time using CUDA events and evaluates the application using different image sizes.

## Objectives

The main objectives of this project are:

1. Apply CUDA parallel programming to image processing.
2. Implement image-processing operations using CUDA kernels.
3. Understand CUDA thread and block organization.
4. Measure GPU execution time.
5. Compare execution behavior across different image sizes and operations.
6. Build a command-line CUDA application that can be compiled and executed in a CUDA-enabled environment.

## GPU Environment

The application was developed using Visual Studio Code and executed in Google Colab using an NVIDIA Tesla T4 GPU.

Test environment:

- GPU: NVIDIA Tesla T4
- Compute Capability: 7.5
- CUDA Toolkit: 12.8
- GPU Memory: approximately 15 GB
- Compiler: `nvcc`

The local development environment was used for writing and organizing the project, while Google Colab provided access to an NVIDIA CUDA-capable GPU for compilation and execution.

## Technologies Used

- C++
- NVIDIA CUDA
- CUDA Runtime API
- `nvcc`
- Google Colab
- Visual Studio Code
- GitHub
- PPM/PGM image formats

## CUDA Implementation

The project contains three CUDA kernels.

### 1. Grayscale Conversion

The grayscale kernel converts an RGB pixel into a single grayscale intensity.

The conversion uses the weighted RGB formula:

```text
Gray = 0.299R + 0.587G + 0.114B
