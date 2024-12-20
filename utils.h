#pragma once
#ifndef UTILS_H
#define UTILS_H

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
};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer();
    ~GpuTimer();

    void Start();
    void Stop();
    float Elapsed();
};

uint32_t swapEndian(uint32_t test);

unsigned char **readImages(const char *fileName, int *numImages, int *numRows, int *numCols);

unsigned char *readLabels(const char *fileName, int *numLabels);

#endif
