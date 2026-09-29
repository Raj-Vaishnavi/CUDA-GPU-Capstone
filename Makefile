NVCC = nvcc

TARGET = cuda_image_processor

SRC = src/main.cu src/image_processing.cu

NVCC_FLAGS = -O2 -std=c++14 -arch=sm_75

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run:
	./$(TARGET) --help
