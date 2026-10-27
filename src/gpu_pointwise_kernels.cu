// gpu_pointwise_kernels.cu
// Stages 1, 3 and 5 on the GPU. Compare each kernel with the CPU function of
// the same stage: the maths is identical. The only change is that the CPU
// version loops over all pixels, while here each thread handles ONE pixel
// and the loop disappears. The GPU runs thousands of threads at once instead.

#include "gpu_kernels.h"

// ---------------------------------------------------------------------------
// How a thread finds "its" pixel (used the same way in every kernel):
//
// We launch a 2D GRID of BLOCKS, and each block is a 2D group of THREADS
// (16 x 16 here). CUDA gives each thread four built-in values:
//   threadIdx.x / threadIdx.y : this thread's position INSIDE its block (0..15)
//   blockIdx.x  / blockIdx.y  : which block this thread belongs to
//   blockDim.x  / blockDim.y  : the size of a block (16 and 16)
// So the pixel column is: (block number * block width) + position in block.
// Example: block 2, thread 5, block width 16  ->  column 2*16 + 5 = 37.
//
// The grid is rounded UP to whole blocks, so a few threads at the right and
// bottom edges land outside the image. They must do nothing, which is what
// the "if (column >= imageWidth ...) return;" check is for. Without it they
// would write past the end of the buffer.
// ---------------------------------------------------------------------------

__global__ void blackLevelKernel(const uint16_t* bayerPixels, float* normalisedBayer,
                                 int imageWidth, int imageHeight,
                                 float blackLevel, float whiteLevel) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    int pixelIndex = row * imageWidth + column;
    float rawValue = static_cast<float>(bayerPixels[pixelIndex]);
    float usefulRange = whiteLevel - blackLevel;
    float normalisedValue = (rawValue - blackLevel) / usefulRange;

    if (normalisedValue < 0.0f) {
        normalisedValue = 0.0f;
    }
    if (normalisedValue > 1.0f) {
        normalisedValue = 1.0f;
    }

    normalisedBayer[pixelIndex] = normalisedValue;
}

__global__ void whiteBalanceKernel(float* rgbImage, int imageWidth, int imageHeight,
                                   float redGain, float greenGain, float blueGain) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    int pixelIndex = row * imageWidth + column;
    int redIndex = pixelIndex * 3 + 0;
    int greenIndex = pixelIndex * 3 + 1;
    int blueIndex = pixelIndex * 3 + 2;

    float redValue = rgbImage[redIndex] * redGain;
    float greenValue = rgbImage[greenIndex] * greenGain;
    float blueValue = rgbImage[blueIndex] * blueGain;

    if (redValue > 1.0f) {
        redValue = 1.0f;
    }
    if (greenValue > 1.0f) {
        greenValue = 1.0f;
    }
    if (blueValue > 1.0f) {
        blueValue = 1.0f;
    }

    // Writing back in place is safe here: each thread touches only its own
    // pixel, so no thread ever reads a value another thread has changed.
    rgbImage[redIndex] = redValue;
    rgbImage[greenIndex] = greenValue;
    rgbImage[blueIndex] = blueValue;
}

// "__device__" marks a helper function that runs on the GPU and can only be
// called from GPU code (from a kernel or another __device__ function).
// Unlike a __global__ kernel, it can return a value.
__device__ unsigned char gammaToByte(float linearValue) {
    if (linearValue < 0.0f) {
        linearValue = 0.0f;
    }
    if (linearValue > 1.0f) {
        linearValue = 1.0f;
    }

    // powf is the float version of pow on the GPU. Its result can differ from
    // the CPU's std::pow in the last tiny digit; Milestone 3 looks at this.
    float gammaValue = powf(linearValue, 1.0f / 2.2f);
    float scaledValue = gammaValue * 255.0f + 0.5f;
    return static_cast<unsigned char>(scaledValue);
}

__global__ void gammaKernel(const float* rgbImage, unsigned char* outputBytes,
                            int imageWidth, int imageHeight) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    // One thread per pixel, so each thread converts all three channels.
    int pixelIndex = row * imageWidth + column;
    for (int channel = 0; channel < 3; channel++) {
        int valueIndex = pixelIndex * 3 + channel;
        outputBytes[valueIndex] = gammaToByte(rgbImage[valueIndex]);
    }
}
