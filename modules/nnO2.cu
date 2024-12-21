#include "../utils/utils.h"

__host__ __device__ float sigmoidKernel3(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ float sigmoidDerivativeKernel3(float x)
{
    return x * (1.0f - x);
}

__global__ void softmaxKernel3(float *x, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float maxVal = -FLT_MAX;
    for (int i = 0; i < size; i++)
        maxVal = max(maxVal, x[i]);

    float expSum = 0.0f;
    for (int i = 0; i < size; i++)
        expSum += expf(x[i] - maxVal);

    x[tid] = expf(x[tid] - maxVal) / expSum;
}

__global__ void calculateCValueKernel3(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

__global__ void updateBiasesKernel3(float *biases, const float *delta, int layerSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < layerSize)
        biases[j] += LEARNING_RATE * delta[j];
}

__global__ void updateWeightsKernel3(float *weights, const float *layer, const float *delta, int layerSize, int prevLayerSize)
{
    extern __shared__ float sharedResult[];
    if (threadIdx.x < prevLayerSize)
        sharedResult[threadIdx.x] = layer[threadIdx.x] * delta[blockIdx.x] * LEARNING_RATE;

    __syncthreads();
    weights[prevLayerSize * blockIdx.x + threadIdx.x] += sharedResult[threadIdx.x];
}

__global__ void createInputLayerKernel3(unsigned char *image, int inputSize, float *inputLayer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize)
        inputLayer[i] = image[i] / 255.0f;
}

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void forwardLayerKernel3(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid) {
    extern __shared__ float sharedResult[];
    if (threadIdx.x < inputSize) {
        sharedResult[threadIdx.x] = inputLayer[threadIdx.x] * weights[inputSize * blockIdx.x + threadIdx.x]; 
    }
    __syncthreads();

    for (int stride = (blockDim.x - 1) / 2 + 1; stride > 0; )
    {
        if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x) {
            sharedResult[threadIdx.x] += sharedResult[threadIdx.x + stride];
        }
        __syncthreads();   

        if (stride == 1)
            break;
        stride = (stride - 1) / 2 + 1;
    }

    if (threadIdx.x == 0)
    {
        if (applySigmoid)
            sharedResult[0] = sigmoidKernel3(sharedResult[0] + biases[blockIdx.x]);
        outputLayer[blockIdx.x] = sharedResult[0];
    }
}

__global__ void calculateDeltaLayerKernel3(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
    extern __shared__ float sharedResult[];
    if (threadIdx.x < nextLayerSize)
        sharedResult[threadIdx.x] = nextLayerDelta[threadIdx.x]  * weights[blockIdx.x * nextLayerSize + threadIdx.x];
    __syncthreads();

    for (int stride = (blockDim.x - 1) / 2 + 1; stride > 0; )
    {
        if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x)
            sharedResult[threadIdx.x] += sharedResult[threadIdx.x + stride];
        __syncthreads();   

        if (stride == 1)
            break;

        stride = (stride - 1) / 2 + 1;
    }

    if (threadIdx.x == 0)
        currentLayerDelta[blockIdx.x] = sharedResult[0] * sigmoidDerivativeKernel3(currentLayer[blockIdx.x]);
}
