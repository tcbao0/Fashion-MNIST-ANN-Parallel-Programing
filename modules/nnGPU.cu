#include "../utils/utils.h"

__host__ __device__ float sigmoidKernel1(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ float sigmoidDerivativeKernel1(float x)
{
    return x * (1.0f - x);
}

__global__ void softmaxKernel1(float *x, int size)
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

__global__ void forwardLayerKernel1(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize)
    {
        float sum = biases[i];
        for (int k = 0; k < inputSize; k++)
            sum += inputLayer[k] * weights[k * outputSize + i];

        if (applySigmoid)
            sum = sigmoidKernel1(sum);
        outputLayer[i] = sum;
    }
}

__global__ void calculateCValueKernel1(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

__global__ void calculateDeltaLayerKernel1(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < currentLayerSize)
    {
        float delta = 0.0f;
        for (int k = 0; k < nextLayerSize; k++)
            delta += nextLayerDelta[k] * weights[j * nextLayerSize + k];

        currentLayerDelta[j] = delta * sigmoidDerivativeKernel1(currentLayer[j]);
    }
}

__global__ void updateWeightsKernel1(float *weights, const float *layer, const float *delta, int layerSize, int prevLayerSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layerSize)
        for (int k = 0; k < prevLayerSize; k++)
            weights[k * layerSize + i] += LEARNING_RATE * layer[k] * delta[i];
}

__global__ void updateBiasesKernel1(float *biases, const float *delta, int layerSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < layerSize)
        biases[j] += LEARNING_RATE * delta[j];
}

__global__ void createInputLayerKernel1(unsigned char *image, int inputSize, float *inputLayer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize)
        inputLayer[i] = image[i] / 255.0f;
}