NVCC = nvcc

TARGET = run

SRCS = main.cu \
       utils.cu \
       modules/CPU/nnCPU.cu \
       modules/GPU/nnGPU.cu

INCLUDE_DIRS = -ICPU -IGPU

NVCC_FLAGS = -std=c++14

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $^ -o $@

clean:
	rm -f $(TARGET)