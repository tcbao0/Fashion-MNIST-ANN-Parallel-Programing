#include "modules/OptimizeV2/nnO2.cu"
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int epochs = 10;
    if (argc == 2) {
        epochs = atoi(argv[1]);
    }

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
    printf("===> Test Set:\n");
    printf("Number of Test images: %d\n", numtestImages);
    printf("Test image size: %dx%d\n", numRows, numCols);
    printf("Number of Test labels: %d\n\n", numtestLabels);

    // Run with host
    printf("---------Training with Host---------\n");
    train(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, epochs);

    // Run with GPU
    printf("\n---------Training with basic GPU---------\n");
    trainKernel1(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, epochs);

    // Run with GPU Optimize 1
    printf("\n---------Training with optimized GPU 1---------\n");
    trainKernel2(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, epochs);

    // Run with GPU Optimize 2
    printf("\n---------Training with optimized GPU 2---------\n");
    trainKernel3(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, epochs);

    // Free memory
    for (int i = 0; i < numTrainImages; i++)
        free(trainImages[i]);

    for (int i = 0; i < numtestImages; i++)
        free(testImages[i]);

    free(trainImages);
    free(testImages);
    free(trainLabels);
    free(testLabels);
    return 0;
}
