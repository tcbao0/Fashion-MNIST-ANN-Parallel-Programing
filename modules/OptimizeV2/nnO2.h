#ifndef NEURAL_NETWORK_OP2_H
#define NEURAL_NETWORK_OP2_H

#include "../CPU/nnCPU.h"

// Not update
__host__ __device__ float sigmoidKernel3(float x);
__host__ __device__ float sigmoidDerivativeKernel3(float x);
__global__ void softmaxKernel3(float *x, int size);
__global__ void createInputLayerKernel3(unsigned char *image, int inputSize, float *inputLayer);
__global__ void calculateCValueKernel3(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i);
__global__ void updateBiasesKernel3(float *biases, const float *delta, int layerSize, float learningRate);
__global__ void updateWeightsKernel3(float *weights, float *layer, float *delta, int layerSize, int prevLayerSize);

// Update
returnStruct trainKernel3(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);
__global__ void forwardLayerKernel3(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid);
__global__ void calculateDeltaLayerKernel3(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);

#endif