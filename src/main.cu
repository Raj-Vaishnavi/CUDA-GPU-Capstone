#include "image_processing.cuh"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
      return 1;                                                            \
    }                                                                      \
  } while (0)

struct Image {
  int width;
  int height;
  int channels;
  std::vector<unsigned char> data;
};


// Reads a binary PPM (P6) RGB image.
bool LoadPPM(const std::string& filename, Image* image) {
  std::ifstream file(filename, std::ios::binary);

  if (!file) {
    std::cerr << "Error: Could not open input file: "
              << filename << std::endl;
    return false;
  }

  std::string format;
  file >> format;

  if (format != "P6") {
    std::cerr << "Error: Only binary PPM (P6) images are supported."
              << std::endl;
    return false;
  }

  file >> image->width >> image->height;

  int max_value;
  file >> max_value;

  if (max_value != 255) {
    std::cerr << "Error: Only 8-bit PPM images are supported."
              << std::endl;
    return false;
  }

  file.get();

  image->channels = 3;

  size_t image_size =
      static_cast<size_t>(image->width) *
      image->height *
      image->channels;

  image->data.resize(image_size);

  file.read(
      reinterpret_cast<char*>(image->data.data()),
      image_size);

  if (!file) {
    std::cerr << "Error: Could not read image data."
              << std::endl;
    return false;
  }

  return true;
}


// Saves an RGB image as a binary PPM file.
bool SavePPM(
    const std::string& filename,
    const Image& image) {

  if (image.channels != 3) {
    std::cerr << "Error: RGB image must have 3 channels."
              << std::endl;
    return false;
  }

  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    std::cerr << "Error: Could not create output file: "
              << filename << std::endl;
    return false;
  }

  file << "P6\n";
  file << image.width << " " << image.height << "\n";
  file << "255\n";

  file.write(
      reinterpret_cast<const char*>(image.data.data()),
      image.data.size());

  return static_cast<bool>(file);
}


// Saves a grayscale image as a binary PGM file.
bool SavePGM(
    const std::string& filename,
    const Image& image) {

  if (image.channels != 1) {
    std::cerr << "Error: Grayscale image must have 1 channel."
              << std::endl;
    return false;
  }

  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    std::cerr << "Error: Could not create output file: "
              << filename << std::endl;
    return false;
  }

  file << "P5\n";
  file << image.width << " " << image.height << "\n";
  file << "255\n";

  file.write(
      reinterpret_cast<const char*>(image.data.data()),
      image.data.size());

  return static_cast<bool>(file);
}


void PrintUsage(const char* program_name) {
  std::cout << "\nCUDA GPU Image Processing\n\n";

  std::cout << "Usage:\n";
  std::cout << "  " << program_name
            << " --input <file> --output <file>"
            << " --operation <operation>\n\n";

  std::cout << "Operations:\n";
  std::cout << "  grayscale\n";
  std::cout << "  brightness\n";
  std::cout << "  blur\n\n";

  std::cout << "Additional option:\n";
  std::cout << "  --brightness <value>   "
               "Brightness adjustment (-255 to 255)\n\n";

  std::cout << "Example:\n";
  std::cout << "  " << program_name
            << " --input input.ppm"
            << " --output output.ppm"
            << " --operation blur\n\n";
}


int main(int argc, char* argv[]) {
  std::string input_file;
  std::string output_file;
  std::string operation;
  int brightness = 50;

  // Parse command-line arguments.
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];

    if (argument == "--input" && i + 1 < argc) {
      input_file = argv[++i];
    } else if (argument == "--output" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (argument == "--operation" && i + 1 < argc) {
      operation = argv[++i];
    } else if (argument == "--brightness" && i + 1 < argc) {
      brightness = std::stoi(argv[++i]);
    } else if (argument == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Error: Unknown or incomplete argument: "
                << argument << std::endl;
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (input_file.empty() ||
      output_file.empty() ||
      operation.empty()) {

    std::cerr << "Error: Missing required arguments."
              << std::endl;

    PrintUsage(argv[0]);
    return 1;
  }

  if (operation != "grayscale" &&
      operation != "brightness" &&
      operation != "blur") {

    std::cerr << "Error: Unsupported operation: "
              << operation << std::endl;

    PrintUsage(argv[0]);
    return 1;
  }

  if (brightness < -255 || brightness > 255) {
    std::cerr << "Error: Brightness must be between "
              << "-255 and 255." << std::endl;
    return 1;
  }

  // Display GPU information.
  int device_count = 0;

  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    std::cerr << "Error: No CUDA-capable GPU was detected."
              << std::endl;
    return 1;
  }

  cudaDeviceProp device_properties;

  CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

  std::cout << "\n========================================\n";
  std::cout << "CUDA GPU Image Processing\n";
  std::cout << "========================================\n";

  std::cout << "GPU: "
            << device_properties.name << std::endl;

  std::cout << "Compute Capability: "
            << device_properties.major << "."
            << device_properties.minor << std::endl;

  std::cout << "Input: " << input_file << std::endl;
  std::cout << "Output: " << output_file << std::endl;
  std::cout << "Operation: " << operation << std::endl;

  // Load input image.
  Image input_image;

  if (!LoadPPM(input_file, &input_image)) {
    return 1;
  }

  std::cout << "Image Size: "
            << input_image.width << " x "
            << input_image.height << std::endl;

  std::cout << "Channels: "
            << input_image.channels << std::endl;

  // Prepare output image.
  Image output_image;

  output_image.width = input_image.width;
  output_image.height = input_image.height;

  if (operation == "grayscale") {
    output_image.channels = 1;

    output_image.data.resize(
        static_cast<size_t>(output_image.width) *
        output_image.height);
  } else {
    output_image.channels = 3;

    output_image.data.resize(
        static_cast<size_t>(output_image.width) *
        output_image.height *
        output_image.channels);
  }

  // Create CUDA events for GPU timing.
  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event));
  // Execute selected CUDA operation.
  if (operation == "grayscale") {

    ConvertToGrayscale(
        input_image.data.data(),
        output_image.data.data(),
        input_image.width,
        input_image.height);

  } else if (operation == "brightness") {

    AdjustBrightness(
        input_image.data.data(),
        output_image.data.data(),
        input_image.width,
        input_image.height,
        brightness);

  } else if (operation == "blur") {

    ApplyBlur(
        input_image.data.data(),
        output_image.data.data(),
        input_image.width,
        input_image.height);
  }

  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float gpu_time = 0.0f;

  CUDA_CHECK(cudaEventElapsedTime(
      &gpu_time,
      start_event,
      stop_event);

  // Save output.
  bool save_success;

  if (operation == "grayscale") {
    save_success = SavePGM(output_file, output_image);
  } else {
    save_success = SavePPM(output_file, output_image);
  }

  if (!save_success) {
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    return 1;
  }

  std::cout << "GPU Processing Time: "
            << gpu_time << " ms" << std::endl;

  std::cout << "Output saved successfully.\n";

  std::cout << "========================================\n";

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  return 0;
}
