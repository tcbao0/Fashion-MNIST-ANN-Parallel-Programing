#ifndef NEURAL_NETWORK_GPU_H
#define NEURAL_NETWORK_GPU_H

#include <cfloat>
#include "../CPU/nnCPU.h"

returnStruct trainKernel1(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols);

__host__ __device__ float sigmoidKernel1(float x);
__host__ __device__ float sigmoidDerivativeKernel1(float x);
__global__ void softmaxKernel1(float *x, float *result, int size);

__global__ void createInputLayerKernel1(unsigned char *image, int inputSize, float *inputLayer);
__global__ void forwardLayerKernel1(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid);
__global__ void calculateCValueKernel1(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int outputSize, int i);
__global__ void calculateDeltaLayerKernel1(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);
__global__ void updateWeightsKernel1(float *weights, float *biases, float *layer, float *delta, int layerSize, int prevLayerSize, float learningRate);
__global__ void updateBiasesKernel1(float *biases, const float *delta, int layerSize, float learningRate);

#endif