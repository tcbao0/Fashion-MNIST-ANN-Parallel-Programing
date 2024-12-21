#ifndef NEURAL_NETWORK_OP1_H
#define NEURAL_NETWORK_OP1_H

#include <cfloat>
#include "../GPU/nnGPU.h"

returnStruct trainKernel2(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);

__global__ void forwardLayerKernel2(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid);
__global__ void calculateCValueKernel2(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int outputSize, int i);
__global__ void calculateDeltaLayerKernel2(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);
__global__ void updateWeightsKernel2(float *weights, float *biases, float *layer, float *delta, int layerSize, int prevLayerSize, float learningRate);
__global__ void updateBiasesKernel2(float *biases, const float *delta, int layerSize, float learningRate);

#endif