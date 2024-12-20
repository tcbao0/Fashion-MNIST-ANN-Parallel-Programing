#include "nnCPU.h"

void forwardLayer(
    float *inputLayer,
    float *weights,
    float *biases,
    float *outputLayer,
    int inputSize,
    int outputSize,
    bool applySigmoid)
{
    for (int j = 0; j < outputSize; j++)
    {
        outputLayer[j] = biases[j];
        for (int k = 0; k < inputSize; k++)
            outputLayer[j] += inputLayer[k] * weights[k * outputSize + j];

        if (applySigmoid)
            outputLayer[j] = sigmoid(outputLayer[j]);
    }
}

void calculateCValue(
    float *outputDelta,
    float *outputLayer,
    unsigned char *trainLabels,
    int outputSize,
    int i)
{
    for (int j = 0; j < outputSize; j++)
        outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

void calculateDeltaLayer(
    float *currentLayer,
    float *nextLayerDelta,
    float *currentLayerDelta,
    float *weights,
    int currentLayerSize,
    int nextLayerSize)
{
    for (int j = 0; j < currentLayerSize; j++)
    {
        currentLayerDelta[j] = 0;
        for (int k = 0; k < nextLayerSize; k++)
            currentLayerDelta[j] += nextLayerDelta[k] * weights[j * nextLayerSize + k];

        currentLayerDelta[j] *= sigmoidDerivative(currentLayer[j]);
    }
}

void updateWeights(
    float *weights,
    float *biases,
    float *layer,
    float *delta,
    int layerSize,
    int prevLayerSize,
    float learningRate)
{
    for (int j = 0; j < layerSize; j++)
        for (int k = 0; k < prevLayerSize; k++)
            weights[k * layerSize + j] += learningRate * layer[k] * delta[j];

    for (int j = 0; j < layerSize; j++)
        biases[j] += learningRate * delta[j];
}

void softmax(float *x, int size)
{
    float maxVal = -FLT_MAX;
    for (int i = 0; i < size; i++)
    {
        maxVal = max(maxVal, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i]);
        sum += x[i];
    }

    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void createInputLayer(unsigned char* image, int inputSize, float** inputLayer) 
{
    *inputLayer = (float*)malloc(inputSize * sizeof(float));
    if (!*inputLayer) {
        printf("Memory allocation failed for inputLayer\n");
        return;
    }
    
    // Chuẩn hóa giá trị pixel (Normalize)
    for (int j = 0; j < inputSize; j++) {
        (*inputLayer)[j] = image[j] / 255.0f;
    }
}


float* initWeightBias(int size)
{
    float* data = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;

    return data;
}

returnStruct train(unsigned char** trainImages, unsigned char* trainLabels, unsigned char** testImages, unsigned char* testLabels, int numTrainImages, int numTestImages, int numRows, int numCols) {
    GpuTimer timer;
    float timeInputLayer = 0, timeHiddenLayer1 = 0, timeHiddenLayer2 = 0, timeOutputLayer = 0;
    float timeInputHidden1 = 0, timeHidden1Hidden2 = 0, timeHidden2Output = 0;
    float finalAccuracy = 0;
    returnStruct result;

    // Allocate
    float* hiddenWeights1 = initWeightBias(INPUT_SIZE * HIDDEN_SIZE_1);
    float* hiddenWeights2 = initWeightBias(HIDDEN_SIZE_1 * HIDDEN_SIZE_2);
    float* outputWeights = initWeightBias(HIDDEN_SIZE_2 * OUTPUT_SIZE);
    float* hiddenBiases1 = initWeightBias(HIDDEN_SIZE_1);
    float* hiddenBiases2 = initWeightBias(HIDDEN_SIZE_2);
    float* outputBiases = initWeightBias(OUTPUT_SIZE);

    // Training
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        for (int batchStart = 0; batchStart < numTrainImages; batchStart += BATCH_SIZE) {
            int batchSize = fmin(BATCH_SIZE, numTrainImages - batchStart);
            for (int i = batchStart; i < batchStart + batchSize; i++) {
                // Xử lý lớp Input và đo thời gian
                timer.Start();
                float* inputLayer= (float*)malloc(INPUT_SIZE * sizeof(float));
                createInputLayer(trainImages[i], INPUT_SIZE, &inputLayer);
                timer.Stop();
                timeInputLayer += timer.Elapsed();
                // Xử lý lớp Hidden 1 và đo thời gian
                timer.Start();
                float* hiddenLayer1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
                forwardLayer(inputLayer, hiddenWeights1, hiddenBiases1, hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1);
                timer.Stop();
                timeHiddenLayer1 += timer.Elapsed();
                // Xử lý lớp Hidden 2 và đo thời gian
                timer.Start();
                float* hiddenLayer2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
                forwardLayer(hiddenLayer1, hiddenWeights2, hiddenBiases2, hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);
                timer.Stop();
                timeHiddenLayer2 += timer.Elapsed();
                // Xử lý lớp Output và đo thời gian
                timer.Start();
                float* outputLayer = (float*)malloc(OUTPUT_SIZE * sizeof(float));
                forwardLayer(hiddenLayer2, outputWeights, outputBiases, outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);
                timer.Stop();
                timeOutputLayer += timer.Elapsed();
                // Áp dụng softmax để có xác suất cho mỗi lớp
                softmax(outputLayer, OUTPUT_SIZE);
                // Quá trình backpropagation
                // Backpropagation để cập nhật trọng số và bias
                float* outputDelta = (float*)malloc(OUTPUT_SIZE * sizeof(float));
                calculateCValue(outputDelta, outputLayer, trainLabels, OUTPUT_SIZE, i);
                // Tính gardient và cập nhật trọng số cho lớp output
                timer.Start();
                updateWeights(outputWeights, outputBiases, hiddenLayer2, outputDelta, OUTPUT_SIZE, HIDDEN_SIZE_2, LEARNING_RATE);
                timer.Stop();
                timeHidden2Output += timer.Elapsed();
                // Tính gardient và cập nhật trọng số cho lớp ẩn 2
                timer.Start();
                float* hiddenLayer2Delta = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
                calculateDeltaLayer(hiddenLayer2, outputDelta, hiddenLayer2Delta, outputWeights, HIDDEN_SIZE_2, OUTPUT_SIZE);
                updateWeights(hiddenWeights2, hiddenBiases2, hiddenLayer1, hiddenLayer2Delta, HIDDEN_SIZE_2, HIDDEN_SIZE_1, LEARNING_RATE);
                timer.Stop();
                timeHidden1Hidden2 += timer.Elapsed();
                // Tính gardient và cập nhật trọng số cho lớp ẩn 1
                timer.Start();
                float* hiddenLayer1Delta = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
                calculateDeltaLayer(hiddenLayer1, hiddenLayer2Delta, hiddenLayer1Delta, hiddenWeights2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);
                updateWeights(hiddenWeights1, hiddenBiases1, inputLayer, hiddenLayer1Delta, HIDDEN_SIZE_1, INPUT_SIZE, LEARNING_RATE);
                timer.Stop();
                timeInputHidden1 += timer.Elapsed();
                free(inputLayer);
                free(hiddenLayer1);
                free(hiddenLayer2);
                free(outputLayer);
                free(outputDelta);
                free(hiddenLayer2Delta);
                free(hiddenLayer1Delta);
            }
        }
        printf("Epoch %d: Accuracy = %.4f\n", epoch, computeFinalAccuracy(testImages, testLabels, numTestImages, numRows, numCols, hiddenWeights1, hiddenWeights2, outputWeights, hiddenBiases1, hiddenBiases2, outputBiases, BATCH_SIZE));
        computeFinalAccuracy(testImages, testLabels, numTestImages, numRows, numCols, hiddenWeights1, hiddenWeights2, outputWeights, hiddenBiases1, hiddenBiases2, outputBiases, BATCH_SIZE);
    }
    free(hiddenWeights1);
    free(hiddenWeights2);
    free(outputWeights);
    free(hiddenBiases1);
    free(hiddenBiases2);
    free(outputBiases);

    result.timeInputLayer = timeInputLayer;
    result.timeHiddenLayer1 = timeHiddenLayer1;
    result.timeHiddenLayer2 = timeHiddenLayer2;
    result.timeOutputLayer = timeOutputLayer;
    result.timeInputHidden1 = timeInputHidden1;
    result.timeHidden1Hidden2 = timeHidden1Hidden2;
    result.timeHidden2Output = timeHidden2Output;
    result.finalAccuracy = finalAccuracy;

    return result;
}


float computeFinalAccuracy(unsigned char** testImages, unsigned char* testLabels, int numTestImages, int numRows, int numCols, float* hiddenWeights1, float* hiddenWeights2, float* outputWeights, float* hiddenBiases1, float* hiddenBiases2, float* outputBiases, int batchSize) {
    int correct = 0;
    for (int i = 0; i < numTestImages; i++) {
        // Quá trình feedforward tương tự như trong training
        float inputLayer[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            inputLayer[j] = testImages[i][j] / 255.0f; // Chuẩn hóa giá trị đầu vào
        }

        // Lớp ẩn 1
        float* hiddenLayer1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
        forwardLayer(inputLayer, hiddenWeights1, hiddenBiases1, hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1);

        float* hiddenLayer2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
        forwardLayer(hiddenLayer1, hiddenWeights2, hiddenBiases2, hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);

        float* outputLayer = (float*)malloc(OUTPUT_SIZE * sizeof(float));
        forwardLayer(hiddenLayer2, outputWeights, outputBiases, outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);

        // Áp dụng softmax để có xác suất cho mỗi lớp
        softmax(outputLayer, OUTPUT_SIZE);

        // Kiểm tra kết quả dự đoán
        int predictedLabel = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (outputLayer[j] > outputLayer[predictedLabel]) {
                predictedLabel = j;
            }
        }

        if (predictedLabel == testLabels[i]) {
            correct++;
        }
    }
    return (float)correct / numTestImages;
}
