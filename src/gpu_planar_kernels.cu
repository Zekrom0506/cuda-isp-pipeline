// gpu_planar_kernels.cu
// Same maths as the naive kernels. The ONLY change is where each colour value
// lives in memory (see gpu_planar_kernels.h). Why it matters:
// in the interleaved layout, 32 neighbouring threads reading "their red value"
// touch addresses 0, 3, 6, 9 ... (every third float), spread over 3x more
// memory than they need. In the planar layout they read addresses 0, 1, 2, 3 ...
// which is one tight, contiguous block: a coalesced access.

#include "gpu_planar_kernels.h"

// Same helpers as in gpu_neighbourhood_kernels.cu. "static" keeps these copies
// private to this file, so their names do not clash with the ones there.
static __device__ int reflectIndexPlanar(int index, int size) {
    if (index < 0) {
        return -index;
    }
    if (index >= size) {
        return 2 * size - 2 - index;
    }
    return index;
}

static __device__ int clampIndexPlanar(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static __device__ float readBayerPlanar(const float* bayerImage, int imageWidth, int imageHeight,
                                        int row, int column) {
    int safeRow = reflectIndexPlanar(row, imageHeight);
    int safeColumn = reflectIndexPlanar(column, imageWidth);
    return bayerImage[safeRow * imageWidth + safeColumn];
}

__global__ void demosaicPlanarKernel(const float* normalisedBayer, float* rgbPlanes,
                                     int imageWidth, int imageHeight) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    float centre = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row, column);
    float up = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row - 1, column);
    float down = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row + 1, column);
    float left = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row, column - 1);
    float right = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row, column + 1);
    float upLeft = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row - 1, column - 1);
    float upRight = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row - 1, column + 1);
    float downLeft = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row + 1, column - 1);
    float downRight = readBayerPlanar(normalisedBayer, imageWidth, imageHeight, row + 1, column + 1);

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

    int pixelCount = imageWidth * imageHeight;
    int pixelIndex = row * imageWidth + column;
    rgbPlanes[pixelIndex] = redValue;
    rgbPlanes[pixelCount + pixelIndex] = greenValue;
    rgbPlanes[2 * pixelCount + pixelIndex] = blueValue;
}

__global__ void whiteBalancePlanarKernel(float* rgbPlanes, int imageWidth, int imageHeight,
                                         float redGain, float greenGain, float blueGain) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    int pixelCount = imageWidth * imageHeight;
    int redIndex = row * imageWidth + column;
    int greenIndex = pixelCount + redIndex;
    int blueIndex = 2 * pixelCount + redIndex;

    float redValue = rgbPlanes[redIndex] * redGain;
    float greenValue = rgbPlanes[greenIndex] * greenGain;
    float blueValue = rgbPlanes[blueIndex] * blueGain;

    if (redValue > 1.0f) {
        redValue = 1.0f;
    }
    if (greenValue > 1.0f) {
        greenValue = 1.0f;
    }
    if (blueValue > 1.0f) {
        blueValue = 1.0f;
    }

    rgbPlanes[redIndex] = redValue;
    rgbPlanes[greenIndex] = greenValue;
    rgbPlanes[blueIndex] = blueValue;
}

__global__ void denoisePlanarKernel(const float* rgbPlanes, float* denoisedPlanes,
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

    int pixelCount = imageWidth * imageHeight;
    int pixelIndex = row * imageWidth + column;

    for (int channel = 0; channel < 3; channel++) {
        // Each channel is its own plane: move to the start of that plane.
        int planeStart = channel * pixelCount;
        float weightedTotal = 0.0f;

        for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int columnOffset = -1; columnOffset <= 1; columnOffset++) {
                int neighbourRow = clampIndexPlanar(row + rowOffset, imageHeight);
                int neighbourColumn = clampIndexPlanar(column + columnOffset, imageWidth);
                int neighbourPixel = neighbourRow * imageWidth + neighbourColumn;
                float neighbourValue = rgbPlanes[planeStart + neighbourPixel];
                float weight = gaussianWeights[rowOffset + 1][columnOffset + 1];
                weightedTotal = weightedTotal + neighbourValue * weight;
            }
        }

        denoisedPlanes[planeStart + pixelIndex] = weightedTotal / weightSum;
    }
}

static __device__ unsigned char gammaToBytePlanar(float linearValue) {
    if (linearValue < 0.0f) {
        linearValue = 0.0f;
    }
    if (linearValue > 1.0f) {
        linearValue = 1.0f;
    }
    float gammaValue = powf(linearValue, 1.0f / 2.2f);
    float scaledValue = gammaValue * 255.0f + 0.5f;
    return static_cast<unsigned char>(scaledValue);
}

__global__ void gammaPlanarKernel(const float* rgbPlanes, unsigned char* outputBytes,
                                  int imageWidth, int imageHeight) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    int pixelCount = imageWidth * imageHeight;
    int pixelIndex = row * imageWidth + column;
    for (int channel = 0; channel < 3; channel++) {
        float linearValue = rgbPlanes[channel * pixelCount + pixelIndex];
        outputBytes[pixelIndex * 3 + channel] = gammaToBytePlanar(linearValue);
    }
}
