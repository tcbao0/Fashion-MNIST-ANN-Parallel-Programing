#ifndef NEURAL_NETWORK_GPU_H
#define NEURAL_NETWORK_GPU_H

#include <cfloat>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../../utils/utils.h"

returnStruct trainKernel1(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);

#endif