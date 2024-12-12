#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define INPUT_SIZE 784
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 128
#define OUTPUT_SIZE 10
#define EPOCHS 10
#define LEARNING_RATE 0.01

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

uint32_t swapEndian(uint32_t test) {
    return ((test & 0xFF) << 24) | 
           ((test & 0xFF00) << 8) | 
           ((test & 0xFF0000) >> 8) | 
           ((test & 0xFF000000) >> 24);
}

unsigned char** readImages(const char* fileName, int* numImages, int* numRows, int* numCols) {
    FILE* f = fopen(fileName, "rb");
	if (f == NULL)
    {
        printf("Failed to open the IDX3-UBYTE file: %s\n", fileName);
        return NULL;
    }

    uint32_t magicNumber, nImages, nRows, nCols;

    fread(&magicNumber, sizeof(uint32_t), 1, f);
    fread(&nImages, sizeof(uint32_t), 1, f);
    fread(&nRows, sizeof(uint32_t), 1, f);
    fread(&nCols, sizeof(uint32_t), 1, f);

    // Chuyển đổi thứ tự byte
    magicNumber = swapEndian(magicNumber);
    nImages = swapEndian(nImages);
    nRows = swapEndian(nRows);
    nCols = swapEndian(nCols);

    *numImages = nImages;
    *numRows = nRows;
    *numCols = nCols;

    unsigned char** images = (unsigned char**)malloc(nImages * sizeof(unsigned char*));
    for (int i = 0; i < nImages; i++) {
        images[i] = (unsigned char*)malloc(nRows * nCols * sizeof(unsigned char));
        fread(images[i], sizeof(unsigned char), nRows * nCols, f);
    }

    fclose(f);
    return images;
}

unsigned char* readLabels(const char* fileName, int* numLabels) {
    FILE* f = fopen(fileName, "rb");
	if (f == NULL)
    {
        printf("Failed to open the IDX1-UBYTE file: %s\n", fileName);
        return NULL;
    }

    uint32_t magicNumber, nLabels;

    fread(&magicNumber, sizeof(uint32_t), 1, f);
    fread(&nLabels, sizeof(uint32_t), 1, f);

    // Chuyển đổi thứ tự byte
    magicNumber = swapEndian(magicNumber);
    nLabels = swapEndian(nLabels);

    *numLabels = nLabels;

    unsigned char* labels = (unsigned char*)malloc(nLabels * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), nLabels, f);

    fclose(f);
    return labels;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

// Hàm softmax cho lớp output
void softmax(float* x, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void forwardLayerCPU(float* inputLayer, float* weights, float* biases, float* outputLayer, int inputSize, int outputSize, bool applySigmoid = true) {
    for (int j = 0; j < outputSize; j++) {
        outputLayer[j] = biases[j];
        for (int k = 0; k < inputSize; k++) {
            outputLayer[j] += inputLayer[k] * weights[k * outputSize + j];
        }
        if (applySigmoid) {
            outputLayer[j] = sigmoid(outputLayer[j]);
        }
    }
}

__global__ void forwardLayerGPU(float* input, float* weights, float* biases, float* output, int inputSize, int outputSize, bool applySigmoid)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < outputSize)
    {
        float sum = biases[tid];
        for (int i = 0; i <inputSize; i++)
        {
            sum += input[i] * weights[i * outputSize + tid];
        }
        output[tid]= applySigmoid ? 1.0f / (1.0f + expf(-sum)) : sum;
    }
}

void calculateCValue(float* outputDelta, float* outputLayer, unsigned char* trainLabels, int outputSize, int i) {
    for (int j = 0; j < outputSize; j++) {
        outputDelta[j] = (trainLabels[i] == j ? 1.0f : 0.0f) - outputLayer[j];
    }
}

void calculateDeltaLayer(float* currentLayer, float* nextLayerDelta, 
                         float* currentLayerDelta, float* weights, 
                         int currentLayerSize, int nextLayerSize) {
    
    // Tính delta cho lớp hiện tại từ delta của lớp tiếp theo
    for (int j = 0; j < currentLayerSize; j++) {
        currentLayerDelta[j] = 0;
        for (int k = 0; k < nextLayerSize; k++) {
            currentLayerDelta[j] += nextLayerDelta[k] * weights[j * nextLayerSize + k];
        }
        currentLayerDelta[j] *= sigmoid_derivative(currentLayer[j]);
    }
}

void updateWeights(float* weights, float* biases, float* layer, float* delta, 
                   int layerSize, int prevLayerSize, float learningRate) {
    // Cập nhật trọng số
    for (int j = 0; j < layerSize; j++) {
        for (int k = 0; k < prevLayerSize; k++) {
            weights[k * layerSize + j] += learningRate * layer[k] * delta[j];
        }
    }
    
    // Cập nhật bias
    for (int j = 0; j < layerSize; j++) {
        biases[j] += learningRate * delta[j];
    }
}

float computeFinalAccuracy(unsigned char** testImages, unsigned char* testLabels, int numTestImages, int numRows, int numCols, float* hiddenWeights1, float* hiddenWeights2, float* outputWeights, float* hiddenBiases1, float* hiddenBiases2, float* outputBiases) {
    int correct = 0;
    for (int i = 0; i < numTestImages; i++) {
        float inputLayer[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            inputLayer[j] = testImages[i][j] / 255.0f; // Chuẩn hóa giá trị đầu vào
        }

        float* hiddenLayer1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
        forwardLayerCPU(inputLayer, hiddenWeights1, hiddenBiases1, hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1);

        float* hiddenLayer2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
        forwardLayerCPU(hiddenLayer1, hiddenWeights2, hiddenBiases2, hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);

        float* outputLayer = (float*)malloc(OUTPUT_SIZE * sizeof(float));
        forwardLayerCPU(hiddenLayer2, outputWeights, outputBiases, outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);

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

int training(unsigned char** trainImages, unsigned char* trainLabels, unsigned char** testImages, unsigned char* testLabels, int numTrainImages, int numTestImages, int numRows, int numCols, int flag) {
    GpuTimer timer;

    // Thời gian xử lý truyền qua từng lớp
    float timeInputLayer = 0, timeHiddenLayer1 = 0, timeHiddenLayer2 = 0, timeOutputLayer = 0;
    // Thời gian cập nhật trọng số mô hình
    float timeInputHidden1 = 0, timeHidden1Hidden2 = 0, timeHidden2Output = 0;
    float finalAccuracy = 0;

    if (flag == 1) {
        // Cấp phát bộ nhớ cho trọng số của các layers
        float* hiddenWeights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float));
        float* hiddenWeights2 = (float*)malloc(HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float));
        float* outputWeights = (float*)malloc(HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float));
        float* hiddenBiases1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
        float* hiddenBiases2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
        float* outputBiases = (float*)malloc(OUTPUT_SIZE * sizeof(float));

        // Thực hiện khởi tạo ngẫu nhiên trọng số ban đầu
        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE_1; i++) hiddenWeights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
        for (int i = 0; i < HIDDEN_SIZE_1 * HIDDEN_SIZE_2; i++) hiddenWeights2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
        for (int i = 0; i < HIDDEN_SIZE_2 * OUTPUT_SIZE; i++) outputWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
        // Gán biases bắt đầu là 0
        for (int i = 0; i < HIDDEN_SIZE_1; i++) hiddenBiases1[i] = 0;
        for (int i = 0; i < HIDDEN_SIZE_2; i++) hiddenBiases2[i] = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++) outputBiases[i] = 0;

        // Duyệt qua từng epoch và huấn luyện mô hình
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            for (int i = 0; i < numTrainImages; i++) {
                // Quá trình feedforward
                // Xử lý lớp Input và đo thời gian
                timer.Start();
                float* inputLayer = (float*)malloc(INPUT_SIZE * sizeof(float));
                for (int j = 0; j < INPUT_SIZE; j++) {
                    inputLayer[j] = trainImages[i][j] / 255.0f; // Normalize
                }
                timer.Stop();
                timeInputLayer += timer.Elapsed();
                // Xử lý lớp Hidden 1 và đo thời gian
                timer.Start();
                float* hiddenLayer1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
                forwardLayerCPU(inputLayer, hiddenWeights1, hiddenBiases1, hiddenLayer1, INPUT_SIZE, HIDDEN_SIZE_1);
                timer.Stop();
                timeHiddenLayer1 += timer.Elapsed();
                // Xử lý lớp Hidden 2 và đo thời gian
                timer.Start();
                float* hiddenLayer2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
                forwardLayerCPU(hiddenLayer1, hiddenWeights2, hiddenBiases2, hiddenLayer2, HIDDEN_SIZE_1, HIDDEN_SIZE_2);
                timer.Stop();
                timeHiddenLayer2 += timer.Elapsed();
                // Xử lý lớp Output và đo thời gian
                timer.Start();
                float* outputLayer = (float*)malloc(OUTPUT_SIZE * sizeof(float));
                forwardLayerCPU(hiddenLayer2, outputWeights, outputBiases, outputLayer, HIDDEN_SIZE_2, OUTPUT_SIZE, false);
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

            printf("Epoch %d: Accuracy = %.4f\n", epoch, computeFinalAccuracy(testImages, testLabels, numTestImages, numRows, numCols, hiddenWeights1, hiddenWeights2, outputWeights, hiddenBiases1, hiddenBiases2, outputBiases));
        }
        finalAccuracy = computeFinalAccuracy(testImages, testLabels, numTestImages, numRows, numCols, hiddenWeights1, hiddenWeights2, outputWeights, hiddenBiases1, hiddenBiases2, outputBiases);

        free(hiddenWeights1);
        free(hiddenWeights2);
        free(outputWeights);
        free(hiddenBiases1); 
        free(hiddenBiases2);
        free(outputBiases);
    }
    else if (flag == 2) {
        // Xử lý với kernel1
        float *d_input, *d_hidden1, *d_hidden2, *d_output;
        float *d_weights1, *d_weights2, *d_weights3;
        float *d_biases1, *d_biases2, *d_biases3;

            //
        CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
        CHECK(cudaMalloc(&d_hidden1, HIDDEN_SIZE_1 * sizeof(float)));
        CHECK(cudaMalloc(&d_hidden2, HIDDEN_SIZE_2 * sizeof(float)));
        CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
        CHECK(cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float)));
        CHECK(cudaMalloc(&d_weights2, INPUT_SIZE * HIDDEN_SIZE_2 * sizeof(float)));
        CHECK(cudaMalloc(&d_weights3, INPUT_SIZE * OUTPUT_SIZE *sizeof(float)));
        CHECK(cudaMalloc(&d_biases1, HIDDEN_SIZE_1 * sizeof(float)));
        CHECK(cudaMalloc(&d_biases2, HIDDEN_SIZE_2 * sizeof(float)));
        CHECK(cudaMalloc(&d_biases3, OUTPUT_SIZE * sizeof(float)));

            //Copy weight to GPU
        CHECK(cudaMemcpy(d_weights1, hiddenWeights1, INPUT_SIZE * HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weights2, hiddenWeights2, INPUT_SIZE * HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weights3, outputWeights, HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_biases1, hiddenBiases1, HIDDEN_SIZE_1 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_biases2, hiddenBiases2, HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_biases3, outputBiases, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));


        for (int epoch = 1; epoch <= EPOCHS; epoch++)
        {
            //TODO
            
            // Xử lý lớp input
            timer.Start();
            for (int i = 0; i < numTrainImages; i++)
            {
                float inputLayer[INPUT_SIZE];
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    inputLayer[j] = trainImages[i][j] / 255.0f;
                }
                cudaMemcpy(d_input, inputLayer, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            }
            timer.Stop();
            timeInputLayer += timer.Elapsed();

            // Xử lý lớp hidden 1
            timer.Start();
            forwardLayerGPU<<<(HIDDEN_SIZE_1 + 255)/ 256, 256>>>(d_input, d_weights1, d_biases1, d_hidden1, INPUT_SIZE, HIDDEN_SIZE_1, true);
            cudaDeviceSynchronize();
            timer.Stop();
            timeHiddenLayer1 += timer.Elapsed();

            // Xử lý lớp hidden 2
            timer.Start();
            forwardLayerGPU<<<(HIDDEN_SIZE_2 + 255) / 256, 256>>>(d_hidden1, d_weights2, d_biases2, d_hidden2, HIDDEN_SIZE_2, true);
            cudaDeviceSynchronize();
            timer.Stop();
            timeHiddenLayer2 += timer.Elapsed();

            // Xử lý lớp output
            timer.Start();
            forwardLayerGPU<<<(OUTPUT_SIZE + 255) / 256, 256>>>(d_hidden2, d_weights3, d_biases3, d_output, OUTPUT_SIZE, false);
            timer.Stop();
            timeOutputLayer += timer.Elapsed();


            float outputLayer[OUTPUT_SIZE];
            cudaMemcpy(outputLayer, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            softmax(outputLayer, OUTPUT_SIZE);
            
        }
        printf("Epoch %d: Accuracy = %.4f\n", epoch, computeFinalAccuracy(testImageFile, testLabels, numTestImages, numRows, numCols, hiddenWeights1, hiddenWeights2, outputWeights, hiddenBiases1, hiddenBiases2, outputBiases));
        cudaFree(d_input);
        cudaFree(d_hidden1);
        cudaFree(d_hidden2);
        cudaFree(d_output);
        cudaFree(d_weights1);
        cudaFree(d_weights2);
        cudaFree(d_weights3);
        cudaFree(d_biases1);
        cudaFree(d_biases2);
        cudaFree(d_biases3);
    } 
    else {
        // Xử lý với kernel2
        for (int i = 1; i <= EPOCHS; i++)
        {
            //TODO
            // Xử lý lớp input
            timer.Start();
            //....
            timer.Stop();
            timeInputLayer += timer.Elapsed();

            // Xử lý lớp hidden 2
            timer.Start();
            //....
            timer.Stop();
            timeHiddenLayer1 += timer.Elapsed();

            // Xử lý lớp hidden 1
            timer.Start();
            //....
            timer.Stop();
            timeHiddenLayer2 += timer.Elapsed();

            // Xử lý lớp output
            timer.Start();
            //....
            timer.Stop();
            timeOutputLayer += timer.Elapsed();
            
            // Tính toán acc và loss
            //....
       
        }
    }
    printf("\nQuá trình feedforward:\n");
    printf("Thời gian chạy trung bình ở lớp Input trong 1 epoch là: %f\n", timeInputLayer / EPOCHS);
    printf("Thời gian chạy trung bình ở lớp Hidden 1 trong 1 epoch là: %f\n", timeHiddenLayer1 / EPOCHS);
    printf("Thời gian chạy trung bình ở lớp Hidden 2 trong 1 epoch là: %f\n", timeHiddenLayer2 / EPOCHS);
    printf("Thời gian chạy trung bình ở lớp Output trong 1 epoch là: %f\n", timeOutputLayer / EPOCHS);

    printf("\nQuá trình backpropagation:\n");
    printf("Thời gian cập nhật trọng số trung bình từ hidden 1 về input trong 1 epoch là: %f\n", timeInputHidden1 / EPOCHS);
    printf("Thời gian cập nhật trọng số trung bình từ hidden 2 về hidden 1 trong 1 epoch là: %f\n", timeHidden1Hidden2 / EPOCHS);
    printf("Thời gian cập nhật trọng số trung bình từ output về hidden 2 trong 1 epoch là: %f\n", timeHidden2Output / EPOCHS);

    printf("Độ chính xác của mô hình trên tập test là: %.2f%%\n", finalAccuracy * 100);

    return 0;
}

int main() {
    const char* trainImageFile = "./dataset/train/train-images-idx3-ubyte";
    const char* trainLabelFile = "./dataset/train/train-labels-idx1-ubyte";
    const char* testImageFile = "./dataset/test/t10k-images-idx3-ubyte";
    const char* testLabelFile = "./dataset/test/t10k-labels-idx1-ubyte";

    int numTrainImages, numtestImages, numRows, numCols;
    int numTrainLabels, numtestLabels;

    // Đọc tập tin ảnh
    unsigned char** trainImages = readImages(trainImageFile, &numTrainImages, &numRows, &numCols);
    unsigned char** testImages = readImages(testImageFile, &numtestImages, &numRows, &numCols);

    if (!trainImages || !testImages) {
        printf("Error reading images file.\n");
        return 1;
    }

    // Đọc tập tin nhãn
    unsigned char* trainLabels = readLabels(trainLabelFile, &numTrainLabels);
    unsigned char* testLabels = readLabels(testLabelFile, &numtestLabels);

    if (!trainLabels || !testLabels) {
        printf("Error reading labels file.\n");
        return 1;
    }

    // Hiển thị thông tin tập huấn luyện
    printf("Training Set:\n");
    printf("Number of training images: %d\n", numTrainImages);
    printf("Training image size: %dx%d\n", numRows, numCols);
    printf("Number of training labels: %d\n", numTrainLabels);

    // Hiển thị thông tin tập kiểm tra
    printf("Test Set:\n");
    printf("Number of Test images: %d\n", numtestImages);
    printf("Test image size: %dx%d\n", numRows, numCols);
    printf("Number of Test labels: %d\n\n", numtestLabels);

    // Khởi chạy với host
    printf("### Huấn luyện mô hình với Host ###\n\n");
    training(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, 1);

    // Khởi chạy với kernel1
    printf("\n### Huấn luyện mô hình với kernel1 ###\n\n");
    training(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, 2);

    // Khởi chạy với kernel2
    printf("\n### Huấn luyện mô hình với kernel2 ###\n\n");
    training(trainImages, trainLabels, testImages, testLabels, numTrainImages, numtestImages, numRows, numCols, 3);

    // Giải phóng bộ nhớ
    for (int i = 0; i < numTrainImages; i++) {
        free(trainImages[i]);
    }
    free(trainImages);

    for (int i = 0; i < numtestImages; i++) {
        free(testImages[i]);
    }
    free(testImages);

    free(trainLabels);
    free(testLabels);

    return 0;
}
