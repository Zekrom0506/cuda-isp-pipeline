// gpu_neighbourhood_kernels.cu
// Stages 2 and 4 on the GPU. Same maths and the same border rules as
// cpu_neighbourhood_stages.cpp (see the comments there for WHY reflect/clamp).
// Naive version: every thread reads its 3x3 neighbours straight from global
// memory (the GPU's large, slow main memory). Neighbouring threads therefore
// read many of the same values again and again. Milestone 6 fixes that.

#include "gpu_kernels.h"

// Same as reflectIndex on the CPU, marked __device__ so GPU code can call it.
__device__ int reflectIndexGpu(int index, int size) {
    if (index < 0) {
        return -index;
    }
    if (index >= size) {
        return 2 * size - 2 - index;
    }
    return index;
}

__device__ int clampIndexGpu(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

__device__ float readBayerValueGpu(const float* bayerImage, int imageWidth, int imageHeight,
                                   int row, int column) {
    int safeRow = reflectIndexGpu(row, imageHeight);
    int safeColumn = reflectIndexGpu(column, imageWidth);
    int pixelIndex = safeRow * imageWidth + safeColumn;
    return bayerImage[pixelIndex];
}

__global__ void demosaicKernel(const float* normalisedBayer, float* rgbImage,
                               int imageWidth, int imageHeight) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    float centre = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row, column);
    float up = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row - 1, column);
    float down = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row + 1, column);
    float left = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row, column - 1);
    float right = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row, column + 1);
    float upLeft = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row - 1, column - 1);
    float upRight = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row - 1, column + 1);
    float downLeft = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row + 1, column - 1);
    float downRight = readBayerValueGpu(normalisedBayer, imageWidth, imageHeight, row + 1, column + 1);

    float crossAverage = (up + down + left + right) / 4.0f;
    float diagonalAverage = (upLeft + upRight + downLeft + downRight) / 4.0f;
    float horizontalAverage = (left + right) / 2.0f;
    float verticalAverage = (up + down) / 2.0f;

    bool isEvenRow = (row % 2 == 0);
    bool isEvenColumn = (column % 2 == 0);

    float redValue = 0.0f;
    float greenValue = 0.0f;
    float blueValue = 0.0f;

    if (isEvenRow && isEvenColumn) {
        redValue = centre;
        greenValue = crossAverage;
        blueValue = diagonalAverage;
    } else if (isEvenRow && !isEvenColumn) {
        redValue = horizontalAverage;
        greenValue = centre;
        blueValue = verticalAverage;
    } else if (!isEvenRow && isEvenColumn) {
        redValue = verticalAverage;
        greenValue = centre;
        blueValue = horizontalAverage;
    } else {
        redValue = diagonalAverage;
        greenValue = crossAverage;
        blueValue = centre;
    }

    int pixelIndex = row * imageWidth + column;
    rgbImage[pixelIndex * 3 + 0] = redValue;
    rgbImage[pixelIndex * 3 + 1] = greenValue;
    rgbImage[pixelIndex * 3 + 2] = blueValue;
}

__global__ void denoiseKernel(const float* rgbImage, float* denoisedImage,
                              int imageWidth, int imageHeight) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    const float gaussianWeights[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };
    const float weightSum = 16.0f;

    int pixelIndex = row * imageWidth + column;

    for (int channel = 0; channel < 3; channel++) {
        float weightedTotal = 0.0f;

        for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int columnOffset = -1; columnOffset <= 1; columnOffset++) {
                int neighbourRow = clampIndexGpu(row + rowOffset, imageHeight);
                int neighbourColumn = clampIndexGpu(column + columnOffset, imageWidth);
                int neighbourPixel = neighbourRow * imageWidth + neighbourColumn;
                float neighbourValue = rgbImage[neighbourPixel * 3 + channel];
                float weight = gaussianWeights[rowOffset + 1][columnOffset + 1];
                weightedTotal = weightedTotal + neighbourValue * weight;
            }
        }

        denoisedImage[pixelIndex * 3 + channel] = weightedTotal / weightSum;
    }
}
