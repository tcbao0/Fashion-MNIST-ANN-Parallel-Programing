#ifndef NEURAL_NETWORK_OP1_H
#define NEURAL_NETWORK_OP1_H

#include <cfloat>
#include "../GPU/nnGPU.h"

returnStruct trainKernel2(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);

#endif