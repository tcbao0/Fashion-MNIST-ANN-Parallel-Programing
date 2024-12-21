NVCC = nvcc

INCLUDES = -Iutils

# Source files
CPU_SRC = CPU.cu
GPU_SRC = GPU.cu
O1_SRC = GPUv1.cu
O2_SRC = GPUv2.cu
UTILS_SRC = utils/utils.cu

OUTPUT_DIR = run
CPU_TARGET = $(OUTPUT_DIR)/run_cpu
GPU_TARGET = $(OUTPUT_DIR)/run_gpu
O1_TARGET = $(OUTPUT_DIR)/run_o1
O2_TARGET = $(OUTPUT_DIR)/run_o2

all: $(OUTPUT_DIR) $(CPU_TARGET) $(GPU_TARGET) $(O1_TARGET) $(O2_TARGET)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(CPU_TARGET): $(CPU_SRC) $(UTILS_SRC)
	$(NVCC) $(INCLUDES) -o $@ $(CPU_SRC) $(UTILS_SRC)

$(GPU_TARGET): $(GPU_SRC) $(UTILS_SRC)
	$(NVCC) $(INCLUDES) -o $@ $(GPU_SRC) $(UTILS_SRC)

$(O1_TARGET): $(O1_SRC) $(UTILS_SRC)
	$(NVCC) $(INCLUDES) -o $@ $(O1_SRC) $(UTILS_SRC)

$(O2_TARGET): $(O2_SRC) $(UTILS_SRC)
	$(NVCC)  $(INCLUDES) -o $@ $(O2_SRC) $(UTILS_SRC)

clean:
	rm -rf $(OUTPUT_DIR)
