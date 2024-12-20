#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cfloat>

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define INPUT_SIZE 784
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 128
#define OUTPUT_SIZE 10
#define EPOCHS 1
#define BATCH_SIZE 512
#define LEARNING_RATE 0.01
#define NSTREAMS 4

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct returnStruct {
    float timeInputLayer, timeHiddenLayer1, timeHiddenLayer2, timeOutputLayer;
    float timeInputHidden1, timeHidden1Hidden2, timeHidden2Output;
    float finalAccuracy;
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

uint32_t swapEndian(uint32_t test)
{
    return ((test & 0xFF) << 24) |
           ((test & 0xFF00) << 8) |
           ((test & 0xFF0000) >> 8) |
           ((test & 0xFF000000) >> 24);
}

unsigned char **readImages(const char *fileName, int *numImages, int *numRows, int *numCols)
{
    FILE *f = fopen(fileName, "rb");
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

    unsigned char **images = (unsigned char **)malloc(nImages * sizeof(unsigned char *));
    for (int i = 0; i < nImages; i++)
    {
        images[i] = (unsigned char *)malloc(nRows * nCols * sizeof(unsigned char));
        fread(images[i], sizeof(unsigned char), nRows * nCols, f);
    }

    fclose(f);
    return images;
}

unsigned char *readLabels(const char *fileName, int *numLabels)
{
    FILE *f = fopen(fileName, "rb");
    if (f == NULL)
    {
        printf("Failed to open the IDX1-UBYTE file: %s\n", fileName);
        return NULL;
    }

    uint32_t magicNumber, nLabels;

    fread(&magicNumber, sizeof(uint32_t), 1, f);
    fread(&nLabels, sizeof(uint32_t), 1, f);

    magicNumber = swapEndian(magicNumber);
    nLabels = swapEndian(nLabels);

    *numLabels = nLabels;
    unsigned char *labels = (unsigned char *)malloc(nLabels * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), nLabels, f);

    fclose(f);
    return labels;
}
