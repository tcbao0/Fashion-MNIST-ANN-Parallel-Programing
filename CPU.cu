#include "utils/train.cu"
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int epochs = 10;
    if (argc == 2) {
        epochs = atoi(argv[1]);
    }

    float timeInputLayer = 0, timeHiddenLayer1 = 0, timeHiddenLayer2 = 0, timeOutputLayer = 0;
    float timeInputHidden1 = 0, timeHidden1Hidden2 = 0, timeHidden2Output = 0;
    float finalAccuracy = 0;
    returnStruct result;

    const char *trainImageFile = "./dataset/train/train-images-idx3-ubyte";
    const char *trainLabelFile = "./dataset/train/train-labels-idx1-ubyte";
    const char *testImageFile = "./dataset/test/t10k-images-idx3-ubyte";
    const char *testLabelFile = "./dataset/test/t10k-labels-idx1-ubyte";

    int numTrainImages, numtestImages, numRows, numCols;
    int numTrainLabels, numtestLabels;

    // Read images file
    unsigned char **trainImages = readImages(trainImageFile, &numTrainImages, &numRows, &numCols);
    unsigned char **testImages = readImages(testImageFile, &numtestImages, &numRows, &numCols);

    if (!trainImages || !testImages)
    {
        printf("Error reading images file.\n");
        return 1;
    }

    // Read labels file
    unsigned char *trainLabels = readLabels(trainLabelFile, &numTrainLabels);
    unsigned char *testLabels = readLabels(testLabelFile, &numtestLabels);

    if (!trainLabels || !testLabels)
    {
        printf("Error reading labels file.\n");
        return 1;
    }

    // Show dataset information of training set
    printf("---------Fashion MNIST Dataset---------\n");
    printf("===> Training Set:\n");
    printf("Number of training images: %d\n", numTrainImages);
    printf("Training image size: %dx%d\n", numRows, numCols);
    printf("Number of training labels: %d\n", numTrainLabels);

    // Show dataset information of testing set
    printf("\n===> Test Set:\n");
    printf("Number of Test images: %d\n", numtestImages);
    printf("Test image size: %dx%d\n", numRows, numCols);
    printf("Number of Test labels: %d\n", numtestLabels);

    // Run with host
    printf("---------Training with Host---------\n");
    result = trainCPU(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, epochs);

    // Free memory
    for (int i = 0; i < numTrainImages; i++)
        free(trainImages[i]);

    for (int i = 0; i < numtestImages; i++)
        free(testImages[i]);

    free(trainImages);
    free(testImages);
    free(trainLabels);
    free(testLabels);

    timeInputLayer = result.timeInputLayer;
    timeHiddenLayer1 = result.timeHiddenLayer1;
    timeHiddenLayer2 = result.timeHiddenLayer2;
    timeOutputLayer = result.timeOutputLayer;
    timeInputHidden1 = result.timeInputHidden1;
    timeHidden1Hidden2 = result.timeHidden1Hidden2;
    timeHidden2Output = result.timeHidden2Output;
    finalAccuracy = result.finalAccuracy;


    printf("---------Summary---------\n");
    printf("Feedforward process:\n");
    printf("Average runtime of the Input layer in one epoch: %f\n", timeInputLayer / epochs);
    printf("Average runtime of Hidden layer 1 in one epoch: %f\n", timeHiddenLayer1 / epochs);
    printf("Average runtime of Hidden layer 2 in one epoch: %f\n", timeHiddenLayer2 / epochs);
    printf("Average runtime of the Output layer in one epoch: %f\n", timeOutputLayer / epochs);

    printf("\nBackpropagation process:\n");
    printf("Average weight update time from Hidden 1 to Input layer in one epoch: %f\n", timeInputHidden1 / epochs);
    printf("Average weight update time from Hidden 2 to Hidden 1 layer in one epoch: %f\n", timeHidden1Hidden2 / epochs);
    printf("Average weight update time from Output to Hidden 2 layer in one epoch: %f\n", timeHidden2Output / epochs);

    printf("Model accuracy on the test set: %.2f%%\n", finalAccuracy * 100);

    return 0;
}
