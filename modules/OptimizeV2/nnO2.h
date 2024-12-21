#ifndef NEURAL_NETWORK_OP2_H
#define NEURAL_NETWORK_OP2_H

#include <cfloat>
#include "../GPU/nnGPU.h"

returnStruct trainKernel3(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);

__global__ void forwardLayerKernel3(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid);
__global__ void calculateDeltaLayerKernel3(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);

#endif