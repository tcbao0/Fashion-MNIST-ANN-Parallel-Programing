NVCC = nvcc

TARGET = run

SRCS = main.cu \
       utils.cu \
       CPU/nnCPU.cu \
       GPU/nnGPU.cu \
       GPUOptimize/nnOptimize.cu

INCLUDE_DIRS = -ICPU -IGPU -IOptimize

NVCC_FLAGS = -std=c++14

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $^ -o $@

clean:
	rm -f $(TARGET)