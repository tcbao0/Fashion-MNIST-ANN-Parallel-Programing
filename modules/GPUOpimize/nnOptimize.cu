#include "nnOptimize.h"

nnOptimize::nnOptimize(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
{
}
nnOptimize::~nnOptimize()
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

__global__ void nnOptimize::forwardLayer(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid = true)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    if (i >= outputSize)
        return;

    int step = (inputSize - 1) / outputSize + 1;
    for (int j = 0; j < step; j++)
        if (i * step + j < inputSize)
            sdata[i * step + j] = inputLayer[i * step + j];
    __syncthreads();

    float sum = biases[i];
    for (int k = 0; k < inputSize; k++)
        sum += sdata[k] * weights[k * outputSize + i];

    if (applySigmoid)
        sum = sigmoid(sum);
    outputLayer[i] = sum;
}

__global__ void nnOptimize::calculateCValue(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int outputSize, int i)
{
}

__global__ void nnOptimize::calculateDeltaLayer(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
}
__global__ void nnOptimize::updateWeights(float *weights, float *biases, float *layer, float *delta, int layerSize, int prevLayerSize, float learningRate)
{
}

__global__ void nnOptimize::computeAccuracy(float *outputLayer, unsigned char *testLabels, int numTestImages, int outputSize, int *correctCount)
{
}
