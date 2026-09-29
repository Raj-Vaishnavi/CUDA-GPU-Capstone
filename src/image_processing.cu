#include "image_processing.cuh"

#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for RGB to grayscale conversion.
__global__ void GrayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixel = y * width + x;
  int rgb_index = pixel * 3;

  unsigned char red = input[rgb_index];
  unsigned char green = input[rgb_index + 1];
  unsigned char blue = input[rgb_index + 2];

  output[pixel] = static_cast<unsigned char>(
      0.299f * red + 0.587f * green + 0.114f * blue);
}


// CUDA kernel for brightness adjustment.
__global__ void BrightnessKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int brightness) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixel = y * width + x;
  int rgb_index = pixel * 3;

  for (int channel = 0; channel < 3; ++channel) {
    int value = input[rgb_index + channel] + brightness;

    if (value > 255) {
      value = 255;
    }

    if (value < 0) {
      value = 0;
    }

    output[rgb_index + channel] =
        static_cast<unsigned char>(value);
  }
}


// CUDA kernel for simple box blur.
__global__ void BlurKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int red_sum = 0;
  int green_sum = 0;
  int blue_sum = 0;
  int count = 0;

  for (int offset_y = -1; offset_y <= 1; ++offset_y) {
    for (int offset_x = -1; offset_x <= 1; ++offset_x) {

      int neighbor_x = x + offset_x;
      int neighbor_y = y + offset_y;

      if (neighbor_x >= 0 &&
          neighbor_x < width &&
          neighbor_y >= 0 &&
          neighbor_y < height) {

        int neighbor_pixel =
            (neighbor_y * width + neighbor_x) * 3;

        red_sum += input[neighbor_pixel];
        green_sum += input[neighbor_pixel + 1];
        blue_sum += input[neighbor_pixel + 2];

        ++count;
      }
    }
  }

  int output_index = (y * width + x) * 3;

  output[output_index] =
      static_cast<unsigned char>(red_sum / count);

  output[output_index + 1] =
      static_cast<unsigned char>(green_sum / count);

  output[output_index + 2] =
      static_cast<unsigned char>(blue_sum / count);
}


void ConvertToGrayscale(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

  unsigned char* device_input = nullptr;
  unsigned char* device_output = nullptr;

  size_t input_size = width * height * 3 * sizeof(unsigned char);
  size_t output_size = width * height * sizeof(unsigned char);

  cudaMalloc(&device_input, input_size);
  cudaMalloc(&device_output, output_size);

  cudaMemcpy(
      device_input,
      input,
      input_size,
      cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 grid_size(
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y);

  GrayscaleKernel<<<grid_size, block_size>>>(
      device_input,
      device_output,
      width,
      height);

  cudaDeviceSynchronize();

  cudaMemcpy(
      output,
      device_output,
      output_size,
      cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);
}


void AdjustBrightness(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int brightness) {

  unsigned char* device_input = nullptr;
  unsigned char* device_output = nullptr;

  size_t image_size =
      width * height * 3 * sizeof(unsigned char);

  cudaMalloc(&device_input, image_size);
  cudaMalloc(&device_output, image_size);

  cudaMemcpy(
      device_input,
      input,
      image_size,
      cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 grid_size(
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y);

  BrightnessKernel<<<grid_size, block_size>>>(
      device_input,
      device_output,
      width,
      height,
      brightness);

  cudaDeviceSynchronize();

  cudaMemcpy(
      output,
      device_output,
      image_size,
      cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);
}


void ApplyBlur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

  unsigned char* device_input = nullptr;
  unsigned char* device_output = nullptr;

  size_t image_size =
      width * height * 3 * sizeof(unsigned char);

  cudaMalloc(&device_input, image_size);
  cudaMalloc(&device_output, image_size);

  cudaMemcpy(
      device_input,
      input,
      image_size,
      cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 grid_size(
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y);

  BlurKernel<<<grid_size, block_size>>>(
      device_input,
      device_output,
      width,
      height);

  cudaDeviceSynchronize();

  cudaMemcpy(
      output,
      device_output,
      image_size,
      cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);
}