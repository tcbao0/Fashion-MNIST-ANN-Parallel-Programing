#include "nnGPU.h"

nnGPU::nnGPU()
{
}

nnGPU::~nnGPU()
{
}

__host__ __device__ float nnGPU::sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ float nnGPU::sigmoidDerivative(float x)
{
    return x * (1.0f - x);
}

__global__ void nnGPU::softmax(float *x, int size)
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

__global__ void nnGPU::forwardLayer(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid = true)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize)
    {
        float sum = biases[i];
        for (int k = 0; k < inputSize; k++)
            sum += inputLayer[k] * weights[k * outputSize + i];

        if (applySigmoid)
            sum = sigmoid(sum);
        outputLayer[i] = sum;
    }
}

__global__ void nnGPU::calculateCValue(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

__global__ void nnGPU::calculateDeltaLayer(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < currentLayerSize)
    {
        float delta = 0.0f;
        for (int k = 0; k < nextLayerSize; k++)
            delta += nextLayerDelta[k] * weights[j * nextLayerSize + k];

        currentLayerDelta[j] = delta * sigmoid_derivative(currentLayer[j]);
    }
}

__global__ void nnGPU::updateWeights(float *weights, const float *layer, const float *delta, int layerSize, int prevLayerSize, float learningRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layerSize)
        for (int k = 0; k < prevLayerSize; k++)
            weights[k * layerSize + i] += learningRate * layer[k] * delta[i];
}
