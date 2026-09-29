#include "image_processing.cuh"

#include <cuda_runtime.h>
#include <iostream>

// Checks CUDA API calls and reports errors clearly.
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
      return;                                                              \
    }                                                                      \
  } while (0)

// Converts an RGB image to grayscale.
__global__ void GrayscaleKernel(const unsigned char* input,
                                unsigned char* output,
                                int width,
                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixelIndex = (y * width + x);
  int rgbIndex = pixelIndex * 3;

  unsigned char red = input[rgbIndex];
  unsigned char green = input[rgbIndex + 1];
  unsigned char blue = input[rgbIndex + 2];

  output[pixelIndex] =
      static_cast<unsigned char>(0.299f * red +
                                 0.587f * green +
                                 0.114f * blue);
}

// Adjusts RGB brightness.
__global__ void BrightnessKernel(const unsigned char* input,
                                 unsigned char* output,
                                 int width,
                                 int height,
                                 int brightness) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixelIndex = (y * width + x);
  int rgbIndex = pixelIndex * 3;

  for (int channel = 0; channel < 3; ++channel) {
    int value = static_cast<int>(input[rgbIndex + channel]) + brightness;

    value = max(0, min(255, value));

    output[rgbIndex + channel] =
        static_cast<unsigned char>(value);
  }
}

// Applies a simple 3x3 blur filter.
__global__ void BlurKernel(const unsigned char* input,
                           unsigned char* output,
                           int width,
                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int redSum = 0;
  int greenSum = 0;
  int blueSum = 0;
  int count = 0;

  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int neighborX = x + dx;
      int neighborY = y + dy;

      if (neighborX >= 0 && neighborX < width &&
          neighborY >= 0 && neighborY < height) {
        int neighborIndex = (neighborY * width + neighborX) * 3;

        redSum += input[neighborIndex];
        greenSum += input[neighborIndex + 1];
        blueSum += input[neighborIndex + 2];

        ++count;
      }
    }
  }

  int outputIndex = (y * width + x) * 3;

  output[outputIndex] =
      static_cast<unsigned char>(redSum / count);
  output[outputIndex + 1] =
      static_cast<unsigned char>(greenSum / count);
  output[outputIndex + 2] =
      static_cast<unsigned char>(blueSum / count);
}

// Converts an RGB image to grayscale using the GPU.
void ConvertToGrayscale(const unsigned char* input,
                        unsigned char* output,
                        int width,
                        int height) {
  unsigned char* deviceInput = nullptr;
  unsigned char* deviceOutput = nullptr;

  size_t inputSize = width * height * 3 * sizeof(unsigned char);
  size_t outputSize = width * height * sizeof(unsigned char);

  CUDA_CHECK(cudaMalloc(&deviceInput, inputSize));
  CUDA_CHECK(cudaMalloc(&deviceOutput, outputSize));

  CUDA_CHECK(cudaMemcpy(deviceInput, input, inputSize,
                        cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  GrayscaleKernel<<<gridSize, blockSize>>>(
      deviceInput, deviceOutput, width, height);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output, deviceOutput, outputSize,
                        cudaMemcpyDeviceToHost));

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
}

// Adjusts brightness using the GPU.
void AdjustBrightness(const unsigned char* input,
                      unsigned char* output,
                      int width,
                      int height,
                      int brightness) {
  unsigned char* deviceInput = nullptr;
  unsigned char* deviceOutput = nullptr;

  size_t imageSize = width * height * 3 * sizeof(unsigned char);

  CUDA_CHECK(cudaMalloc(&deviceInput, imageSize));
  CUDA_CHECK(cudaMalloc(&deviceOutput, imageSize));

  CUDA_CHECK(cudaMemcpy(deviceInput, input, imageSize,
                        cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  BrightnessKernel<<<gridSize, blockSize>>>(
      deviceInput, deviceOutput, width, height, brightness);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output, deviceOutput, imageSize,
                        cudaMemcpyDeviceToHost));

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
}

// Applies a simple blur using the GPU.
void ApplyBlur(const unsigned char* input,
               unsigned char* output,
               int width,
               int height) {
  unsigned char* deviceInput = nullptr;
  unsigned char* deviceOutput = nullptr;

  size_t imageSize = width * height * 3 * sizeof(unsigned char);

  CUDA_CHECK(cudaMalloc(&deviceInput, imageSize));
  CUDA_CHECK(cudaMalloc(&deviceOutput, imageSize));

  CUDA_CHECK(cudaMemcpy(deviceInput, input, imageSize,
                        cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  BlurKernel<<<gridSize, blockSize>>>(
      deviceInput, deviceOutput, width, height);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output, deviceOutput, imageSize,
                        cudaMemcpyDeviceToHost));

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
}
