// gpu_shared_denoise.cu
// Denoise with shared memory.
//
// The problem with the naive denoise: each pixel reads its 9 neighbours from
// global memory, and each of those values is ALSO read by the 8 surrounding
// threads. So every value is requested from global memory about 9 times.
//
// The fix: each block first copies its patch of the image (its 32x8 pixels plus
// a 1-pixel border, 34x10 values) into SHARED MEMORY, once. Then every thread
// reads its 9 neighbours from shared memory instead.
//
// Shared memory is a small, very fast memory INSIDE each GPU core (SM). It is
// shared by all threads of ONE block, and it only exists while that block runs.
// Global memory is large (15 GB on the T4) but far away and much slower.

#include "gpu_shared_denoise.h"

static __device__ int clampIndexShared(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

__global__ void denoiseSharedKernel(const float* rgbPlanes, float* denoisedPlanes,
                                    int imageWidth, int imageHeight) {
    // "__shared__" puts this array in shared memory. There is ONE copy per
    // block, seen by all of the block's threads (unlike normal local variables,
    // where every thread has its own private copy).
    // 3 channels x 10 rows x 34 columns x 4 bytes = 4080 bytes per block.
    __shared__ float sharedTile[3][SHARED_TILE_HEIGHT][SHARED_TILE_WIDTH];

    int pixelCount = imageWidth * imageHeight;

    // Top-left image position of this block's 32x8 area.
    int blockStartColumn = blockIdx.x * TILE_WIDTH;
    int blockStartRow = blockIdx.y * TILE_HEIGHT;

    // ---- Phase 1: the whole block loads the tile together ----
    // The tile has 34 x 10 = 340 values per channel, but the block has only
    // 32 x 8 = 256 threads. So we number the threads 0..255 and let thread t
    // load values t, t + 256, t + 512, ... until all 340 are loaded.
    // Consecutive threads load consecutive columns, so the loads are coalesced.
    int threadNumber = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int threadsPerBlock = TILE_WIDTH * TILE_HEIGHT;
    int valuesPerTile = SHARED_TILE_WIDTH * SHARED_TILE_HEIGHT;

    for (int loadNumber = threadNumber; loadNumber < valuesPerTile; loadNumber = loadNumber + threadsPerBlock) {
        int tileRow = loadNumber / SHARED_TILE_WIDTH;
        int tileColumn = loadNumber % SHARED_TILE_WIDTH;

        // Tile position 0 is one pixel ABOVE / LEFT of the block (the halo),
        // hence "- HALO_SIZE". Clamp so halo pixels outside the image use the
        // nearest edge pixel, exactly like the CPU version.
        int imageRow = clampIndexShared(blockStartRow + tileRow - HALO_SIZE, imageHeight);
        int imageColumn = clampIndexShared(blockStartColumn + tileColumn - HALO_SIZE, imageWidth);
        int imagePixel = imageRow * imageWidth + imageColumn;

        for (int channel = 0; channel < 3; channel++) {
            sharedTile[channel][tileRow][tileColumn] = rgbPlanes[channel * pixelCount + imagePixel];
        }
    }

    // "__syncthreads()" is a barrier: every thread of the block waits here
    // until ALL threads of the block have reached it. Without it, a fast thread
    // could start reading neighbours that a slower thread has not loaded yet.
    //
    // IMPORTANT: every thread must reach __syncthreads, including threads
    // whose pixel is outside the image (they still helped load the tile). That
    // is why the "outside the image? return" check comes AFTER this line.
    // If some threads returned early, the others would wait here forever.
    __syncthreads();

    // ---- Phase 2: each thread computes its own pixel from shared memory ----
    int column = blockStartColumn + threadIdx.x;
    int row = blockStartRow + threadIdx.y;
    if (column >= imageWidth || row >= imageHeight) {
        return;
    }

    const float gaussianWeights[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };
    const float weightSum = 16.0f;

    // This thread's pixel sits at (threadIdx.y + 1, threadIdx.x + 1) in the
    // tile, because of the 1-pixel halo above and to the left.
    int centreTileRow = threadIdx.y + HALO_SIZE;
    int centreTileColumn = threadIdx.x + HALO_SIZE;
    int pixelIndex = row * imageWidth + column;

    for (int channel = 0; channel < 3; channel++) {
        float weightedTotal = 0.0f;

        // Same loop order as the naive kernel, so the sum is added up in the
        // same order and the result should be bit-for-bit identical.
        for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int columnOffset = -1; columnOffset <= 1; columnOffset++) {
                float neighbourValue = sharedTile[channel][centreTileRow + rowOffset][centreTileColumn + columnOffset];
                float weight = gaussianWeights[rowOffset + 1][columnOffset + 1];
                weightedTotal = weightedTotal + neighbourValue * weight;
            }
        }

        denoisedPlanes[channel * pixelCount + pixelIndex] = weightedTotal / weightSum;
    }
}
