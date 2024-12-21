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
    printf("Number of Test labels: %d\n\n", numtestLabels);

    // Run with GPU Optimize 2
    printf("\n---------Training with optimized GPU 2---------\n");
    result = trainKernel3(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, epochs);

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

    printf("\nQuá trình feedforward:\n");
    printf("Thời gian chạy trung bình ở lớp Input trong 1 epoch là: %f\n", timeInputLayer / epochs);
    printf("Thời gian chạy trung bình ở lớp Hidden 1 trong 1 epoch là: %f\n", timeHiddenLayer1 / epochs);
    printf("Thời gian chạy trung bình ở lớp Hidden 2 trong 1 epoch là: %f\n", timeHiddenLayer2 / epochs);
    printf("Thời gian chạy trung bình ở lớp Output trong 1 epoch là: %f\n", timeOutputLayer / epochs);

    printf("\nQuá trình backpropagation:\n");
    printf("Thời gian cập nhật trọng số trung bình từ hidden 1 về input trong 1 epoch là: %f\n", timeInputHidden1 / epochs);
    printf("Thời gian cập nhật trọng số trung bình từ hidden 2 về hidden 1 trong 1 epoch là: %f\n", timeHidden1Hidden2 / epochs);
    printf("Thời gian cập nhật trọng số trung bình từ output về hidden 2 trong 1 epoch là: %f\n", timeHidden2Output / epochs);

    printf("Độ chính xác của mô hình trên tập test là: %.2f%%\n", finalAccuracy * 100);

    return 0;
}
