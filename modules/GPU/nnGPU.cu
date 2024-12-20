#include "nnGPU.h"

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

__global__ void forwardLayerKernel1(float *inputLayer, float *weights, float *biases, float *outputLayer, int inputSize, int outputSize, bool applySigmoid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize)
    {
        float sum = biases[i];
        for (int k = 0; k < inputSize; k++)
            sum += inputLayer[k] * weights[k * outputSize + i];

        if (applySigmoid)
            sum = sigmoidKernel1(sum);
        outputLayer[i] = sum;
    }
}

__global__ void calculateCValueKernel1(float *outputDelta, float *outputLayer, unsigned char *trainLabels, int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
}

__global__ void calculateDeltaLayerKernel1(float *currentLayer, float *nextLayerDelta, float *currentLayerDelta, float *weights, int currentLayerSize, int nextLayerSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < currentLayerSize)
    {
        float delta = 0.0f;
        for (int k = 0; k < nextLayerSize; k++)
            delta += nextLayerDelta[k] * weights[j * nextLayerSize + k];

        currentLayerDelta[j] = delta * sigmoidDerivativeKernel1(currentLayer[j]);
    }
}

__global__ void updateWeightsKernel1(float *weights, const float *layer, const float *delta, int layerSize, int prevLayerSize, float learningRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layerSize)
        for (int k = 0; k < prevLayerSize; k++)
            weights[k * layerSize + i] += learningRate * layer[k] * delta[i];
}

__global__ void updateBiasesKernel1(float* biases, const float* delta, int layerSize, float learningRate) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (j < layerSize) {
        biases[j] += learningRate * delta[j];
    }
}


__global__ void createInputLayerKernel1 (unsigned char* image, int inputSize, float* inputLayer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inputSize)
        inputLayer[i] = image[i] / 255.0f;
}

returnStruct train(unsigned char** trainImages, unsigned char* trainLabels, unsigned char** testImages, unsigned char* testLabels, int numTrainImages, int numTestImages, int numRows, int numCols) {
    GpuTimer timer;
    float timeInputLayer = 0, timeHiddenLayer1 = 0, timeHiddenLayer2 = 0, timeOutputLayer = 0;
    float timeInputHidden1 = 0, timeHidden1Hidden2 = 0, timeHidden2Output = 0;
    float finalAccuracy = 0;
    returnStruct result;

    // Khởi tạo trọng số cho layers trong GPU
    float* d_hiddenWeights1, *d_hiddenWeights2, *d_outputWeights;
    float* d_hiddenBiases1, *d_hiddenBiases2, *d_outputBiases;
    unsigned char *d_trainImages, *d_testImages, *d_trainLabels, *d_testLabels;

    // Phân bổ bộ nhớ trên GPU
    CHECK(cudaMalloc((void**)&d_hiddenWeights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_hiddenWeights2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_outputWeights, HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_hiddenBiases1, HIDDEN_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_hiddenBiases2, HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_outputBiases, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_trainImages, numTrainImages * INPUT_SIZE * sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&d_trainLabels, numTrainImages * sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&d_testImages, numTestImages * INPUT_SIZE * sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&d_testLabels, numTestImages * sizeof(unsigned char)));

    // Tạo mảng trên CPU để khởi tạo dữ liệu
    float* h_hiddenWeights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float));
    float* h_hiddenWeights2 = (float*)malloc(HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float));
    float* h_outputWeights = (float*)malloc(HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float));
    float* h_hiddenBiases1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
    float* h_hiddenBiases2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
    float* h_outputBiases = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Khởi tạo dữ liệu ngẫu nhiên trên CPU
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE_1; i++) h_hiddenWeights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < HIDDEN_SIZE_1 * HIDDEN_SIZE_2; i++) h_hiddenWeights2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < HIDDEN_SIZE_2 * OUTPUT_SIZE; i++) h_outputWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < HIDDEN_SIZE_1; i++) h_hiddenBiases1[i] = 0;
    for (int i = 0; i < HIDDEN_SIZE_2; i++) h_hiddenBiases2[i] = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) h_outputBiases[i] = 0;

    // Sao chép dữ liệu từ CPU sang GPU
    CHECK(cudaMemcpy(d_hiddenWeights1, h_hiddenWeights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hiddenWeights2, h_hiddenWeights2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_outputWeights, h_outputWeights, HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hiddenBiases1, h_hiddenBiases1, HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hiddenBiases2, h_hiddenBiases2, HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_outputBiases, h_outputBiases, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));     
    CHECK(cudaMemcpy(d_trainLabels, trainLabels, numTrainImages * sizeof(unsigned char), cudaMemcpyHostToDevice));  
    CHECK(cudaMemcpy(d_testLabels, testLabels, numTestImages * sizeof(unsigned char), cudaMemcpyHostToDevice));  

    // Chuyển dữ liệu train thành 1 chiều
    unsigned char* h_trainImages = (unsigned char*)malloc(numTrainImages * INPUT_SIZE * sizeof(unsigned char));
    unsigned char* h_testImages = (unsigned char*)malloc(numTestImages * INPUT_SIZE * sizeof(unsigned char));

    for (int i = 0;i < numTrainImages; i++)
    {
        for (int j = 0;j < INPUT_SIZE; j++)
        {
            h_trainImages[i * INPUT_SIZE + j] = trainImages[i][j];
        }
    }

    for (int i = 0;i < numTestImages; i++)
    {
        for (int j = 0;j < INPUT_SIZE; j++)
        {
            h_testImages[i * INPUT_SIZE + j] = testImages[i][j];
        }
    }

    timer.Start();
    CHECK(cudaMemcpy(d_trainImages, h_trainImages, numTrainImages * INPUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));  
    timer.Stop();
    timeInputLayer += timer.Elapsed();
    CHECK(cudaMemcpy(d_testImages, h_testImages, numTestImages * INPUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice)); 

    dim3 blockSize;
    dim3 gridSize;

    // Khai báo và cấp phát đầu ra trong mạng neural ở GPU
    float* d_inputLayer, *d_hiddenLayer1, *d_hiddenLayer2, *d_outputLayer, *d_outputDelta, *d_hiddenLayer2Delta, *d_hiddenLayer1Delta;

    CHECK(cudaMalloc((void**)&d_inputLayer, INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_hiddenLayer1, HIDDEN_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_hiddenLayer2, HIDDEN_SIZE_2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_outputLayer, OUTPUT_SIZE * sizeof(float))); 
    CHECK(cudaMalloc((void**)&d_outputDelta, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_hiddenLayer2Delta, HIDDEN_SIZE_2 * sizeof(float)));    
    CHECK(cudaMalloc((void**)&d_hiddenLayer1Delta, HIDDEN_SIZE_1 * sizeof(float)));      

    // Duyệt qua từng epoch và huấn luyện mô hình
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        for (int batchStart = 0; batchStart < numTrainImages; batchStart += BATCH_SIZE) {
            int batchSize = fmin(BATCH_SIZE, numTrainImages - batchStart);

            // Quá trình feedforward
            for (int i = batchStart; i < batchStart + batchSize; i++) {
            
                // Quá trình feedforward
                // Xử lý lớp Input và đo thời gian
                timer.Start();
                blockSize=dim3(128);
                gridSize = ((INPUT_SIZE - 1)/blockSize.x + 1);
                createInputLayerKernel1<<<gridSize, blockSize >>>(d_trainImages + i * INPUT_SIZE, INPUT_SIZE, d_inputLayer);
                timer.Stop();
                timeInputLayer += timer.Elapsed();

                // Xử lý lớp Hidden 1 và đo thời gian
                timer.Start();
                blockSize=dim3(128);
                gridSize = ((HIDDEN_SIZE_1 -1)/blockSize.x + 1);
                forwardLayerKernel1<<<gridSize, blockSize >>>(d_inputLayer, d_hiddenWeights1, d_hiddenBiases1, d_hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1, true);
                timer.Stop();
                timeHiddenLayer1 += timer.Elapsed();

                // Xử lý lớp Hidden 2 và đo thời gian
                timer.Start();
                blockSize=dim3(128);
                gridSize = ((HIDDEN_SIZE_2 -1)/blockSize.x + 1);
                forwardLayerKernel1<<<gridSize, blockSize >>>(d_hiddenLayer1, d_hiddenWeights2, d_hiddenBiases2, d_hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2, true);
                timer.Stop();
                timeHiddenLayer2 += timer.Elapsed();

                // Xử lý lớp Output và đo thời gian
                timer.Start();
                blockSize=dim3(128);
                gridSize = ((OUTPUT_SIZE -1)/blockSize.x + 1);
                forwardLayerKernel1<<<gridSize, blockSize >>>(d_hiddenLayer2, d_outputWeights, d_outputBiases, d_outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);

                // Áp dụng softmax để có xác suất cho mỗi lớp
                blockSize=dim3(OUTPUT_SIZE);
                gridSize = ((OUTPUT_SIZE -1)/blockSize.x + 1);
                
                softmaxKernel1<<<gridSize, blockSize >>>(d_outputLayer, OUTPUT_SIZE);
                timer.Stop();
                timeOutputLayer += timer.Elapsed();

                // Quá trình backpropagation
                // Backpropagation để cập nhật trọng số và bias
                timer.Start();          
                blockSize=dim3(1);
                gridSize = ((OUTPUT_SIZE-1)/blockSize.x + 1);
                calculateCValueKernel1<<<gridSize, blockSize>>>(d_outputDelta, d_outputLayer, d_trainLabels, i);

                // Tính gardient và cập nhật trọng số cho lớp output
                blockSize=dim3(128);
                gridSize = ((OUTPUT_SIZE -1)/blockSize.x + 1);
                updateWeightsKernel1<<<gridSize, blockSize>>>(d_outputWeights, d_hiddenLayer2, d_outputDelta, OUTPUT_SIZE, HIDDEN_SIZE_2, LEARNING_RATE);
                blockSize=dim3(128);
                gridSize = ((OUTPUT_SIZE -1)/blockSize.x + 1);
                updateBiasesKernel1<<<gridSize, blockSize>>>(d_outputBiases, d_outputDelta, OUTPUT_SIZE, LEARNING_RATE);
                timer.Stop();
                timeHidden2Output += timer.Elapsed();

                // Tính gardient và cập nhật trọng số cho lớp ẩn 2
                timer.Start();
                blockSize=dim3(128);
                gridSize=((HIDDEN_SIZE_2 -1)/blockSize.x + 1);
                calculateDeltaLayerKernel1<<<gridSize, blockSize>>>(d_hiddenLayer2, d_outputDelta, d_hiddenLayer2Delta, d_outputWeights, HIDDEN_SIZE_2, OUTPUT_SIZE);

                blockSize=dim3(128);
                gridSize=((HIDDEN_SIZE_2 -1)/blockSize.x + 1);
                updateWeightsKernel1<<<gridSize, blockSize>>>(d_hiddenWeights2, d_hiddenLayer1, d_hiddenLayer2Delta, HIDDEN_SIZE_2, HIDDEN_SIZE_1, LEARNING_RATE);

                blockSize=dim3(128);
                gridSize=((HIDDEN_SIZE_2 -1)/blockSize.x + 1);
                updateBiasesKernel1<<<gridSize, blockSize>>>(d_hiddenBiases2, d_hiddenLayer2Delta, HIDDEN_SIZE_2, LEARNING_RATE);

                timer.Stop();
                timeHidden1Hidden2 += timer.Elapsed();

                // Tính gardient và cập nhật trọng số cho lớp ẩn 1
                timer.Start();                   
                blockSize=dim3(128);
                gridSize=((HIDDEN_SIZE_1 -1)/blockSize.x + 1);
                calculateDeltaLayerKernel1<<<gridSize, blockSize>>>(d_hiddenLayer1, d_hiddenLayer2Delta, d_hiddenLayer1Delta, d_hiddenWeights2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);

                blockSize=dim3(128);
                gridSize=((HIDDEN_SIZE_1 -1)/blockSize.x + 1);
                updateWeightsKernel1<<<gridSize, blockSize>>>(d_hiddenWeights1, d_inputLayer, d_hiddenLayer1Delta, HIDDEN_SIZE_1, INPUT_SIZE, LEARNING_RATE);

                blockSize=dim3(128);
                gridSize=((HIDDEN_SIZE_1 -1)/blockSize.x + 1);
                updateBiasesKernel1<<<gridSize, blockSize>>>(d_hiddenBiases1, d_hiddenLayer1Delta, HIDDEN_SIZE_1, LEARNING_RATE);

                // timer.Stop();
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