#ifndef NEURAL_NETWORK_OPTIMIZE_H
#define NEURAL_NETWORK_OPTIMIZE_H

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>

class nnOptimize
{
public:
    nnOptimize(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize);
    ~nnOptimize();

    void forwardLayer(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid = true);
    void calculateCValue(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int outputSize, int i);
    void calculateDeltaLayer(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);
    void updateWeights(float *weights, float *biases, float *layer, float *delta, int layerSize, int prevLayerSize, float learningRate);
    void computeAccuracy(float *outputLayer, unsigned char *testLabels, int numTestImages, int outputSize, int *correctCount);

private:
    int inputSize, hiddenSize1, hiddenSize2, outputSize;

    __host__ __device__ float sigmoid(float x);
    __host__ __device__ float sigmoidDerivative(float x);
    __global__ void softmax(float *x, float *result, int size);
}

#endif