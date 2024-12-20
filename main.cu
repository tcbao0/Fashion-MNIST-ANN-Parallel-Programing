#include "utils/utils.h"
#include "modules/CPU/nnCPU.h"
#include "modules/GPU/nnGPU.h"

int main()
{
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
    printf("Training Set:\n");
    printf("Number of training images: %d\n", numTrainImages);
    printf("Training image size: %dx%d\n", numRows, numCols);
    printf("Number of training labels: %d\n", numTrainLabels);

    // Show dataset information of testing set
    printf("Test Set:\n");
    printf("Number of Test images: %d\n", numtestImages);
    printf("Test image size: %dx%d\n", numRows, numCols);
    printf("Number of Test labels: %d\n\n", numtestLabels);

    // Run with host
    printf("### Training with Host ###\n\n");
    train(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols);

    // Run with GPU
    printf("\n### Training with basic GPU ###\n\n");
    trainKernel1(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols);

    // Run with optimized GPU version 1
    // printf("\n### Training with optimized GPU version 1 ###\n\n");
    // nnOptimizeVer1 nnDeivceOp1();
    // nnDeivceOp1.train(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols);

    // Run with optimized GPU version 2
    // printf("\n### Training with optimized GPU version 2 ###\n\n");
    // nnOptimizeVer2 nnDeivceOp2();
    // nnDeivceOp2.train(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols);

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
