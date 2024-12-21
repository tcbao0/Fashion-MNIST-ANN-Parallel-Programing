#include "../utils/utils.h"

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float sigmoidDerivative(float x)
{
    return x * (1.0f - x);
}

void forwardLayer(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid)
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

void calculateCValue(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int outputSize, int i)
{
    for (int j = 0; j < outputSize; j++)
        outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

void calculateDeltaLayer(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
    for (int j = 0; j < currentLayerSize; j++)
    {
        currentLayerDelta[j] = 0;
        for (int k = 0; k < nextLayerSize; k++)
            currentLayerDelta[j] += nextLayerDelta[k] * weights[j * nextLayerSize + k];

        currentLayerDelta[j] *= sigmoidDerivative(currentLayer[j]);
    }
}

void updateWeights(float *weights, float *biases, float *layer, float *delta, int layerSize, int prevLayerSize, float learningRate)
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

void createInputLayer(unsigned char *image, int inputSize, float **inputLayer)
{
    *inputLayer = (float *)malloc(inputSize * sizeof(float));
    if (!*inputLayer)
    {
        printf("Memory allocation failed for inputLayer\n");
        return;
    }

    for (int j = 0; j < inputSize; j++)
        (*inputLayer)[j] = image[j] / 255.0f;
}

float *initWeightBias(int size)
{
    float *data = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;

    return data;
}

float computeFinalAccuracy(unsigned char **testImages, unsigned char *testLabels, int numTestImages, int numRows, int numCols, float *hiddenWeights1, float *hiddenWeights2, float *outputWeights, float *hiddenBiases1, float *hiddenBiases2, float *outputBiases, int batchSize)
{
    int correct = 0;
    for (int i = 0; i < numTestImages; i++)
    {
        // Feedforward
        float inputLayer[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++)
            inputLayer[j] = testImages[i][j] / 255.0f;

        // Lớp ẩn 1
        float *hiddenLayer1 = (float *)malloc(HIDDEN_SIZE_1 * sizeof(float));
        forwardLayer(inputLayer, hiddenWeights1, hiddenBiases1, hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1);

        float *hiddenLayer2 = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
        forwardLayer(hiddenLayer1, hiddenWeights2, hiddenBiases2, hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);

        float *outputLayer = (float *)malloc(OUTPUT_SIZE * sizeof(float));
        forwardLayer(hiddenLayer2, outputWeights, outputBiases, outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);

        // Softmax
        softmax(outputLayer, OUTPUT_SIZE);

        // Check predict
        int predictedLabel = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++)
            if (outputLayer[j] > outputLayer[predictedLabel])
                predictedLabel = j;

        if (predictedLabel == testLabels[i])
            correct++;
    }
    return (float)correct / numTestImages;
}
