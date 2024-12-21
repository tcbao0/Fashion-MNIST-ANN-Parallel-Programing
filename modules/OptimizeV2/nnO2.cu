#include "nnO2.h"

__host__ __device__ float sigmoidKernel1(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ float sigmoidDerivativeKernel1(float x)
{
    return x * (1.0f - x);
}

__global__ void softmaxKernel1(float *x, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float maxVal = -FLT_MAX;
    for (int i = 0; i < size; i++)
        maxVal = max(maxVal, x[i]);

    float expSum = 0.0f;
    for (int i = 0; i < size; i++)
        expSum += expf(x[i] - maxVal);

    x[tid] = expf(x[tid] - maxVal) / expSum;
}

__global__ void forwardLayerKernel3(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid) {
    extern __shared__ float sharedResult[];
    if (threadIdx.x < inputSize) {
        sharedResult[threadIdx.x] = inputLayer[threadIdx.x] * weights[inputSize * blockIdx.x + threadIdx.x]; 
    }
    __syncthreads();

    for (int stride = (blockDim.x - 1) / 2 + 1; stride > 0; )
    {
        if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x) {
            sharedResult[threadIdx.x] += sharedResult[threadIdx.x + stride];
        }
        __syncthreads();   

        if (stride == 1)
            break;
        stride = (stride - 1) / 2 + 1;
    }

    if (threadIdx.x == 0)
    {
        if (applySigmoid)
            sharedResult[0] = sigmoidKernel1(sharedResult[0] + biases[blockIdx.x]);
        outputLayer[blockIdx.x] = sharedResult[0];
    }
}

__global__ void calculateCValueKernel1(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

__global__ void calculateDeltaLayerKernel3(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
    extern __shared__ float sharedResult[];
    if (threadIdx.x < nextLayerSize)
        sharedResult[threadIdx.x] = nextLayerDelta[threadIdx.x]  * weights[blockIdx.x * nextLayerSize + threadIdx.x];
    __syncthreads();

    for (int stride = (blockDim.x - 1) / 2 + 1; stride > 0; )
    {
        if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x)
            sharedResult[threadIdx.x] += sharedResult[threadIdx.x + stride];
        __syncthreads();   

        if (stride == 1)
            break;

        stride = (stride - 1) / 2 + 1;
    }

    if (threadIdx.x == 0)
        currentLayerDelta[blockIdx.x] = sharedResult[0] * sigmoidDerivativeKernel1(currentLayer[blockIdx.x]);
}

__global__ void updateWeightsKernel1(float *weights, const float *layer, const float *delta, int layerSize, int prevLayerSize, float learningRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layerSize)
        for (int k = 0; k < prevLayerSize; k++)
            weights[k * layerSize + i] += learningRate * layer[k] * delta[i];
}

__global__ void updateBiasesKernel1(float *biases, const float *delta, int layerSize, float learningRate)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < layerSize)
        biases[j] += learningRate * delta[j];
}

__global__ void createInputLayerKernel1(unsigned char *image, int inputSize, float *inputLayer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize)
        inputLayer[i] = image[i] / 255.0f;
}

returnStruct trainKernel3(unsigned char **trainImages, unsigned char *trainLabels, unsigned char **testImages, unsigned char *testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int numEpochs) {
    GpuTimer timer;
    float timeInputLayer = 0, timeHiddenLayer1 = 0, timeHiddenLayer2 = 0, timeOutputLayer = 0;
    float timeInputHidden1 = 0, timeHidden1Hidden2 = 0, timeHidden2Output = 0;
    float finalAccuracy = 0;
    returnStruct result;

    // Initialize
    float *d_hiddenWeights1, *d_hiddenWeights2, *d_outputWeights;
    float *d_hiddenBiases1, *d_hiddenBiases2, *d_outputBiases;
    unsigned char *d_trainImages, *d_testImages, *d_trainLabels, *d_testLabels;

    // Allocate memory on GPU
    CHECK(cudaMalloc((void **)&d_hiddenWeights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenWeights2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_outputWeights, HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenBiases1, HIDDEN_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenBiases2, HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_outputBiases, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_trainImages, numTrainImages * INPUT_SIZE * sizeof(unsigned char)));
    CHECK(cudaMalloc((void **)&d_trainLabels, numTrainImages * sizeof(unsigned char)));
    CHECK(cudaMalloc((void **)&d_testImages, numTestImages * INPUT_SIZE * sizeof(unsigned char)));
    CHECK(cudaMalloc((void **)&d_testLabels, numTestImages * sizeof(unsigned char)));
    float *h_hiddenWeights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float));
    float *h_hiddenWeights2 = (float *)malloc(HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float));
    float *h_outputWeights = (float *)malloc(HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float));
    float *h_hiddenBiases1 = (float *)malloc(HIDDEN_SIZE_1 * sizeof(float));
    float *h_hiddenBiases2 = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
    float *h_outputBiases = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE_1; i++)
        h_hiddenWeights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < HIDDEN_SIZE_1 * HIDDEN_SIZE_2; i++)
        h_hiddenWeights2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < HIDDEN_SIZE_2 * OUTPUT_SIZE; i++)
        h_outputWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < HIDDEN_SIZE_1; i++)
        h_hiddenBiases1[i] = 0;
    for (int i = 0; i < HIDDEN_SIZE_2; i++)
        h_hiddenBiases2[i] = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        h_outputBiases[i] = 0;

    // Copy data to device
    CHECK(cudaMemcpy(d_hiddenWeights1, h_hiddenWeights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hiddenWeights2, h_hiddenWeights2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_outputWeights, h_outputWeights, HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hiddenBiases1, h_hiddenBiases1, HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hiddenBiases2, h_hiddenBiases2, HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_outputBiases, h_outputBiases, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_trainLabels, trainLabels, numTrainImages * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_testLabels, testLabels, numTestImages * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Flatten images
    unsigned char *h_trainImages = (unsigned char *)malloc(numTrainImages * INPUT_SIZE * sizeof(unsigned char));
    unsigned char *h_testImages = (unsigned char *)malloc(numTestImages * INPUT_SIZE * sizeof(unsigned char));

    for (int i = 0; i < numTrainImages; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            h_trainImages[i * INPUT_SIZE + j] = trainImages[i][j];

    for (int i = 0; i < numTestImages; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            h_testImages[i * INPUT_SIZE + j] = testImages[i][j];

    CHECK(cudaMemcpy(d_trainImages, h_trainImages, numTrainImages * INPUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_testImages, h_testImages, numTestImages * INPUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 blockSize;
    dim3 gridSize;
    int sharedMemorySize;

    // Allocate memory for layers
    float *d_inputLayer, *d_hiddenLayer1, *d_hiddenLayer2, *d_outputLayer, *d_outputDelta, *d_hiddenLayer2Delta, *d_hiddenLayer1Delta;

    CHECK(cudaMalloc((void **)&d_inputLayer, INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenLayer1, HIDDEN_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenLayer2, HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_outputLayer, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_outputDelta, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenLayer2Delta, HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_hiddenLayer1Delta, HIDDEN_SIZE_1 * sizeof(float)));

    // Training
    for (int epoch = 1; epoch <= numEpochs; epoch++)
    {
        for (int batchStart = 0; batchStart < numTrainImages; batchStart += BATCH_SIZE)
        {
            int batchSize = fmin(BATCH_SIZE, numTrainImages - batchStart);
            for (int i = batchStart; i < batchStart + batchSize; i++)
            {
                // Forward pass
                // Input
                timer.Start();
                blockSize = dim3(128);
                gridSize = ((INPUT_SIZE - 1) / blockSize.x + 1);
                createInputLayerKernel1<<<gridSize, blockSize>>>(d_trainImages + i * INPUT_SIZE, INPUT_SIZE, d_inputLayer);
                timer.Stop();
                timeInputLayer += timer.Elapsed();

                // Hidden 1
                timer.Start();
                blockSize = dim3(INPUT_SIZE);
                gridSize = ((HIDDEN_SIZE_1 * INPUT_SIZE - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * INPUT_SIZE;
                forwardLayerKernel3<<<gridSize, blockSize, sharedMemorySize>>>(d_inputLayer, d_hiddenWeights1, d_hiddenBiases1, d_hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1, true);
                timer.Stop();
                timeHiddenLayer1 += timer.Elapsed();

                // Hidden 2
                timer.Start();
                blockSize = dim3(HIDDEN_SIZE_1);
                gridSize = ((HIDDEN_SIZE_2 * HIDDEN_SIZE_1 - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * HIDDEN_SIZE_1;
                forwardLayerKernel3<<<gridSize, blockSize, sharedMemorySize>>>(d_hiddenLayer1, d_hiddenWeights2, d_hiddenBiases2, d_hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2, true);
                timer.Stop();
                timeHiddenLayer2 += timer.Elapsed();

                // Output
                timer.Start();
                blockSize = dim3(HIDDEN_SIZE_2);
                gridSize = ((OUTPUT_SIZE * HIDDEN_SIZE_2 - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * HIDDEN_SIZE_2;
                forwardLayerKernel3<<<gridSize, blockSize, sharedMemorySize>>>(d_hiddenLayer2, d_outputWeights, d_outputBiases, d_outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);

                // Softmax
                blockSize = dim3(OUTPUT_SIZE);
                gridSize = ((OUTPUT_SIZE - 1) / blockSize.x + 1);
                softmaxKernel1<<<gridSize, blockSize>>>(d_outputLayer, OUTPUT_SIZE);
                timer.Stop();
                timeOutputLayer += timer.Elapsed();

                // Backpropagation
                // Backpropagation and update weights and biases
                timer.Start();
                blockSize = dim3(1);
                gridSize = ((OUTPUT_SIZE - 1) / blockSize.x + 1);
                calculateCValueKernel1<<<gridSize, blockSize>>>(d_outputDelta, d_outputLayer, d_trainLabels, i);

                // Gradient and update weights for output layer
                blockSize = dim3(HIDDEN_SIZE_2);
                gridSize = ((OUTPUT_SIZE * HIDDEN_SIZE_2 - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * HIDDEN_SIZE_2;
                updateWeightsKernel2<<<gridSize, blockSize, sharedMemorySize>>>(d_outputWeights, d_hiddenLayer2, d_outputDelta, OUTPUT_SIZE, HIDDEN_SIZE_2);
                blockSize = dim3(128);
                gridSize = ((OUTPUT_SIZE - 1) / blockSize.x + 1);
                updateBiasesKernel1<<<gridSize, blockSize>>>(d_outputBiases, d_outputDelta, OUTPUT_SIZE, LEARNING_RATE);
                timer.Stop();
                timeHidden2Output += timer.Elapsed();

                // Gradient and update weights for hidden layer 2
                timer.Start();
                blockSize = dim3(OUTPUT_SIZE);
                gridSize = ((HIDDEN_SIZE_2 * OUTPUT_SIZE - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * OUTPUT_SIZE;
                calculateDeltaLayerKernel3<<<gridSize, blockSize, sharedMemorySize>>>(d_hiddenLayer2, d_outputDelta, d_hiddenLayer2Delta, d_outputWeights, HIDDEN_SIZE_2, OUTPUT_SIZE);

                blockSize = dim3(HIDDEN_SIZE_1);
                gridSize = ((HIDDEN_SIZE_1 * HIDDEN_SIZE_2 - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * HIDDEN_SIZE_1;
                updateWeightsKernel2<<<gridSize, blockSize, sharedMemorySize>>>(d_hiddenWeights2, d_hiddenLayer1, d_hiddenLayer2Delta, HIDDEN_SIZE_2, HIDDEN_SIZE_1);

                blockSize = dim3(128);
                gridSize = ((HIDDEN_SIZE_2 - 1) / blockSize.x + 1);
                updateBiasesKernel1<<<gridSize, blockSize>>>(d_hiddenBiases2, d_hiddenLayer2Delta, HIDDEN_SIZE_2, LEARNING_RATE);
                timer.Stop();
                timeHidden1Hidden2 += timer.Elapsed();

                // Gradient and update weights for hidden layer 1
                timer.Start();
                blockSize = dim3(HIDDEN_SIZE_2);
                gridSize = ((HIDDEN_SIZE_1 * HIDDEN_SIZE_2 - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * HIDDEN_SIZE_2;
                calculateDeltaLayerKernel3<<<gridSize, blockSize, sharedMemorySize>>>(d_hiddenLayer1, d_hiddenLayer2Delta, d_hiddenLayer1Delta, d_hiddenWeights2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);

                blockSize = dim3(INPUT_SIZE);
                gridSize = ((INPUT_SIZE * HIDDEN_SIZE_1 - 1) / blockSize.x + 1);
                sharedMemorySize = sizeof(float) * INPUT_SIZE;
                updateWeightsKernel2<<<gridSize, blockSize, sharedMemorySize>>>(d_hiddenWeights1, d_inputLayer, d_hiddenLayer1Delta, HIDDEN_SIZE_1, INPUT_SIZE);

                blockSize = dim3(128);
                gridSize = ((HIDDEN_SIZE_1 - 1) / blockSize.x + 1);
                updateBiasesKernel1<<<gridSize, blockSize>>>(d_hiddenBiases1, d_hiddenLayer1Delta, HIDDEN_SIZE_1, LEARNING_RATE);
                timer.Stop();
                timeInputHidden1 += timer.Elapsed();
            }
        }

        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Copy result from device memory
        CHECK(cudaMemcpy(h_hiddenWeights1, d_hiddenWeights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_hiddenWeights2, d_hiddenWeights2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_outputWeights, d_outputWeights, HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_hiddenBiases1, d_hiddenBiases1, HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_hiddenBiases2, d_hiddenBiases2, HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_outputBiases, d_outputBiases, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        printf("Epoch %d: Accuracy = %.4f\n", epoch, computeFinalAccuracy(testImages, testLabels, numTestImages, numRows, numCols, h_hiddenWeights1, h_hiddenWeights2, h_outputWeights, h_hiddenBiases1, h_hiddenBiases2, h_outputBiases, BATCH_SIZE));
    }
    finalAccuracy = computeFinalAccuracy(testImages, testLabels, numTestImages, numRows, numCols, h_hiddenWeights1, h_hiddenWeights2, h_outputWeights, h_hiddenBiases1, h_hiddenBiases2, h_outputBiases, BATCH_SIZE);

    free(h_hiddenWeights1);
    free(h_hiddenWeights2);
    free(h_outputWeights);
    free(h_hiddenBiases1);
    free(h_hiddenBiases2);
    free(h_outputBiases);
    free(h_trainImages);
    free(h_testImages);

    CHECK(cudaFree(d_hiddenWeights1));
    CHECK(cudaFree(d_hiddenWeights2));
    CHECK(cudaFree(d_outputWeights));
    CHECK(cudaFree(d_hiddenBiases1));
    CHECK(cudaFree(d_hiddenBiases2));
    CHECK(cudaFree(d_outputBiases));
    CHECK(cudaFree(d_trainImages));
    CHECK(cudaFree(d_trainLabels));
    CHECK(cudaFree(d_testImages));
    CHECK(cudaFree(d_testLabels));
    CHECK(cudaFree(d_inputLayer));
    CHECK(cudaFree(d_hiddenLayer1));
    CHECK(cudaFree(d_hiddenLayer2));
    CHECK(cudaFree(d_outputLayer));
    CHECK(cudaFree(d_outputDelta));
    CHECK(cudaFree(d_hiddenLayer2Delta));
    CHECK(cudaFree(d_hiddenLayer1Delta));

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
