#ifndef NEURAL_NETWORK_OP1_H
#define NEURAL_NETWORK_OP1_H

#include "../CPU/nnCPU.h"

// Not update
__host__ __device__ float sigmoidKernel2(float x);
__host__ __device__ float sigmoidDerivativeKernel2(float x);
__global__ void softmaxKernel2(float *x, int size);
__global__ void createInputLayerKernel2(unsigned char *image, int inputSize, float *inputLayer);
__global__ void calculateCValueKernel2(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i);
__global__ void updateBiasesKernel2(float *biases, const float *delta, int layerSize, float learningRate);

// Update
returnStruct trainKernel2(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);
__global__ void forwardLayerKernel2(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid);
__global__ void calculateDeltaLayerKernel2(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);
__global__ void updateWeightsKernel2(float *weights, float *layer, float *delta, int layerSize, int prevLayerSize);

#endif