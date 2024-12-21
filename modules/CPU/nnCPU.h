#ifndef NEURAL_NETWORK_CPU_H
#define NEURAL_NETWORK_CPU_H

#include "../../utils/utils.h"

returnStruct train(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);

float sigmoid(float x);
float sigmoidDerivative(float x);
void softmax(float *x, int size);

void createInputLayer(unsigned char *image, int inputSize, float **inputLayer);
void forwardLayer(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid = true);
void calculateCValue(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int outputSize, int i);
void calculateDeltaLayer(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize);
void updateWeights(float *weights, float *biases, float *layer, float *delta, int layerSize, int prevLayerSize, float learningRate);

float *initWeightBias(int size);
void computeAccuracy(float *outputLayer, unsigned char *testLabels, int numTestImages, int outputSize, int *correctCount);
float computeFinalAccuracy(unsigned char **testImages, unsigned char *testLabels, int numTestImages, int numRows, int numCols, float *hiddenWeights1, float *hiddenWeights2, float *outputWeights, float *hiddenBiases1, float *hiddenBiases2, float *outputBiases, int batchSize);

#endif