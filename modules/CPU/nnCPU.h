#ifndef NEURAL_NETWORK_CPU_H
#define NEURAL_NETWORK_CPU_H

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../../utils/utils.h"

returnStruct train(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs);

#endif