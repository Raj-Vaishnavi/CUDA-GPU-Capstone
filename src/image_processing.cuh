#ifndef IMAGE_PROCESSING_CUH_
#define IMAGE_PROCESSING_CUH_

#include <cuda_runtime.h>

// Converts an RGB image to grayscale using the GPU.
void ConvertToGrayscale(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height);

// Adjusts the brightness of an RGB image using the GPU.
void AdjustBrightness(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int brightness);

// Applies a simple blur to an RGB image using the GPU.
void ApplyBlur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height);

#endif  // IMAGE_PROCESSING_CUH_