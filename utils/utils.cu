#include "utils.h"

GpuTimer::GpuTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GpuTimer::~GpuTimer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GpuTimer::Start()
{
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
}

void GpuTimer::Stop()
{
    cudaEventRecord(stop, 0);
}

float GpuTimer::Elapsed()
{
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
}

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