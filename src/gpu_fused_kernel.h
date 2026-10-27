// gpu_fused_kernel.h
// Milestone 7: all 5 ISP stages in ONE kernel.

#pragma once

#include <cstdint>

// Each block produces a 32 x 8 patch of output pixels (one per thread).
const int FUSED_TILE_WIDTH = 32;
const int FUSED_TILE_HEIGHT = 8;

// Denoise needs RGB values 1 pixel around the patch, and demosaicing THOSE
// needs Bayer values 1 pixel further out. So the two shared tiles are:
//   RGB tile:   halo 1  ->  34 x 10
//   Bayer tile: halo 2  ->  36 x 12
const int RGB_HALO = 1;
const int BAYER_HALO = 2;
const int RGB_TILE_WIDTH = FUSED_TILE_WIDTH + 2 * RGB_HALO;       // 34
const int RGB_TILE_HEIGHT = FUSED_TILE_HEIGHT + 2 * RGB_HALO;     // 10
const int BAYER_TILE_WIDTH = FUSED_TILE_WIDTH + 2 * BAYER_HALO;   // 36
const int BAYER_TILE_HEIGHT = FUSED_TILE_HEIGHT + 2 * BAYER_HALO; // 12

__global__ void fusedPipelineKernel(const uint16_t* bayerPixels, unsigned char* outputBytes,
                                    int imageWidth, int imageHeight,
                                    float blackLevel, float whiteLevel,
                                    float redGain, float greenGain, float blueGain);
