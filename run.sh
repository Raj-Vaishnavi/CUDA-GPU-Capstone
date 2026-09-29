#!/bin/bash

set -e

echo "========================================"
echo "CUDA GPU Image Processing"
echo "========================================"

echo ""
echo "Compiling project..."
make

echo ""
echo "Compilation completed successfully."

echo ""
echo "Displaying program usage:"
./cuda_image_processor --help

echo ""
echo "Project is ready for image processing."