// gpu_shared_denoise.h
// Milestone 6: the denoise stage using shared memory (planar layout).

#pragma once

// The block size MUST match the tile size, because each thread computes
// exactly one output pixel of the tile. We use the best shape from Milestone 5.
const int TILE_WIDTH = 32;
const int TILE_HEIGHT = 8;

// The tile in shared memory is 1 pixel bigger on every side (the "halo"),
// because the pixels on the edge of the tile need neighbours from outside it.
const int HALO_SIZE = 1;
const int SHARED_TILE_WIDTH = TILE_WIDTH + 2 * HALO_SIZE;    // 34
const int SHARED_TILE_HEIGHT = TILE_HEIGHT + 2 * HALO_SIZE;  // 10

__global__ void denoiseSharedKernel(const float* rgbPlanes, float* denoisedPlanes,
                                    int imageWidth, int imageHeight);
