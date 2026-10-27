// gpu_fused_kernel.cu
// All 5 stages in one kernel ("kernel fusion").
//
// Why: in the 5-kernel pipeline, every stage writes its full-size result to
// global memory and the next stage reads it straight back. Those in-between
// images (up to 270 MB each) exist only to pass data from one kernel to the
// next. Here, the in-between values live in shared memory and registers
// instead, and global memory sees only the raw input (read) and the final
// 8-bit output (written).
//
// The kernel works in three phases, separated by __syncthreads():
//   Phase 1: load Bayer tile (36 x 12), applying stage 1 (black level) on the way.
//   Phase 2: build RGB tile (34 x 10) with stage 2 (demosaic) and stage 3 (white balance).
//   Phase 3: each thread does stage 4 (denoise) and stage 5 (gamma) for its own
//            pixel and writes 3 bytes to global memory.
//
// The trade-off: the 1-pixel border of each RGB tile is also inside the
// neighbouring blocks' tiles, so those pixels are demosaiced twice (by both
// blocks). 34 x 10 = 340 demosaics for 256 output pixels: about 33% extra maths
// in exchange for far less memory traffic.

#include "gpu_fused_kernel.h"

static __device__ int reflectIndexFused(int index, int size) {
    if (index < 0) {
        return -index;
    }
    if (index >= size) {
        return 2 * size - 2 - index;
    }
    return index;
}

static __device__ int clampIndexFused(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

// Reflect first (correct Bayer colour at the image edge, as on the CPU), then
// clamp. The clamp only matters for tile positions far outside the image in
// the last block of a row/column; their values are never used, but the index
// must still stay inside the buffer.
static __device__ int safeBayerIndex(int index, int size) {
    int reflectedIndex = reflectIndexFused(index, size);
    return clampIndexFused(reflectedIndex, size);
}

static __device__ unsigned char gammaToByteFused(float linearValue) {
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

static __device__ float clampToOne(float value) {
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

__global__ void fusedPipelineKernel(const uint16_t* bayerPixels, unsigned char* outputBytes,
                                    int imageWidth, int imageHeight,
                                    float blackLevel, float whiteLevel,
                                    float redGain, float greenGain, float blueGain) {
    __shared__ float bayerTile[BAYER_TILE_HEIGHT][BAYER_TILE_WIDTH];
    __shared__ float rgbTile[3][RGB_TILE_HEIGHT][RGB_TILE_WIDTH];

    int blockStartColumn = blockIdx.x * FUSED_TILE_WIDTH;
    int blockStartRow = blockIdx.y * FUSED_TILE_HEIGHT;
    int threadNumber = threadIdx.y * FUSED_TILE_WIDTH + threadIdx.x;
    int threadsPerBlock = FUSED_TILE_WIDTH * FUSED_TILE_HEIGHT;

    // Image position of tile position (0, 0) for each tile.
    int bayerTileOriginRow = blockStartRow - BAYER_HALO;
    int bayerTileOriginColumn = blockStartColumn - BAYER_HALO;
    int rgbTileOriginRow = blockStartRow - RGB_HALO;
    int rgbTileOriginColumn = blockStartColumn - RGB_HALO;

    // ---- Phase 1: load the Bayer tile, applying black level correction ----
    // Same "thread t loads t, t+256, t+512 ..." pattern as Milestone 6.
    float usefulRange = whiteLevel - blackLevel;
    int bayerTileSize = BAYER_TILE_WIDTH * BAYER_TILE_HEIGHT;
    for (int loadNumber = threadNumber; loadNumber < bayerTileSize; loadNumber = loadNumber + threadsPerBlock) {
        int tileRow = loadNumber / BAYER_TILE_WIDTH;
        int tileColumn = loadNumber % BAYER_TILE_WIDTH;
        int imageRow = safeBayerIndex(bayerTileOriginRow + tileRow, imageHeight);
        int imageColumn = safeBayerIndex(bayerTileOriginColumn + tileColumn, imageWidth);

        float rawValue = static_cast<float>(bayerPixels[imageRow * imageWidth + imageColumn]);
        float normalisedValue = (rawValue - blackLevel) / usefulRange;
        if (normalisedValue < 0.0f) {
            normalisedValue = 0.0f;
        }
        if (normalisedValue > 1.0f) {
            normalisedValue = 1.0f;
        }
        bayerTile[tileRow][tileColumn] = normalisedValue;
    }
    __syncthreads();

    // ---- Phase 2: demosaic + white balance into the RGB tile ----
    int rgbTileSize = RGB_TILE_WIDTH * RGB_TILE_HEIGHT;
    for (int workNumber = threadNumber; workNumber < rgbTileSize; workNumber = workNumber + threadsPerBlock) {
        int tileRow = workNumber / RGB_TILE_WIDTH;
        int tileColumn = workNumber % RGB_TILE_WIDTH;

        // The denoise stage clamps at the image edge, so an RGB position
        // outside the image must hold the demosaiced value of the nearest
        // edge pixel. We demosaic THAT pixel.
        int imageRow = clampIndexFused(rgbTileOriginRow + tileRow, imageHeight);
        int imageColumn = clampIndexFused(rgbTileOriginColumn + tileColumn, imageWidth);

        // Where that pixel sits in the Bayer tile. Phase 1 stored reflected
        // edge values in the halo, so neighbours at +-1 are always available.
        int centreRow = imageRow - bayerTileOriginRow;
        int centreColumn = imageColumn - bayerTileOriginColumn;

        float centre = bayerTile[centreRow][centreColumn];
        float up = bayerTile[centreRow - 1][centreColumn];
        float down = bayerTile[centreRow + 1][centreColumn];
        float left = bayerTile[centreRow][centreColumn - 1];
        float right = bayerTile[centreRow][centreColumn + 1];
        float upLeft = bayerTile[centreRow - 1][centreColumn - 1];
        float upRight = bayerTile[centreRow - 1][centreColumn + 1];
        float downLeft = bayerTile[centreRow + 1][centreColumn - 1];
        float downRight = bayerTile[centreRow + 1][centreColumn + 1];

        float crossAverage = (up + down + left + right) / 4.0f;
        float diagonalAverage = (upLeft + upRight + downLeft + downRight) / 4.0f;
        float horizontalAverage = (left + right) / 2.0f;
        float verticalAverage = (up + down) / 2.0f;

        // Colour is decided by the IMAGE position (RGGB), not the tile position.
        bool isEvenRow = (imageRow % 2 == 0);
        bool isEvenColumn = (imageColumn % 2 == 0);

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

        // Stage 3, white balance, applied straight away.
        rgbTile[0][tileRow][tileColumn] = clampToOne(redValue * redGain);
        rgbTile[1][tileRow][tileColumn] = clampToOne(greenValue * greenGain);
        rgbTile[2][tileRow][tileColumn] = clampToOne(blueValue * blueGain);
    }
    __syncthreads();

    // ---- Phase 3: denoise + gamma for this thread's own pixel ----
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

    int centreTileRow = threadIdx.y + RGB_HALO;
    int centreTileColumn = threadIdx.x + RGB_HALO;
    int pixelIndex = row * imageWidth + column;

    for (int channel = 0; channel < 3; channel++) {
        float weightedTotal = 0.0f;
        for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int columnOffset = -1; columnOffset <= 1; columnOffset++) {
                float neighbourValue = rgbTile[channel][centreTileRow + rowOffset][centreTileColumn + columnOffset];
                float weight = gaussianWeights[rowOffset + 1][columnOffset + 1];
                weightedTotal = weightedTotal + neighbourValue * weight;
            }
        }
        float denoisedValue = weightedTotal / weightSum;
        outputBytes[pixelIndex * 3 + channel] = gammaToByteFused(denoisedValue);
    }
}
