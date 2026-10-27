// gpu_planar_kernels.h
// Stages 2 to 5 rewritten for a PLANAR colour layout (Milestone 5).
//
// Interleaved layout (Milestones 1-4):  R G B R G B R G B ...
// Planar layout (this file):            R R R ... then G G G ... then B B B ...
//   red   of pixel p is at rgbPlanes[p]
//   green of pixel p is at rgbPlanes[pixelCount + p]
//   blue  of pixel p is at rgbPlanes[2 * pixelCount + p]
// It is still ONE buffer of the same size; only the order of values changes.
// Stage 1 is unchanged (the Bayer image has one value per pixel either way).

#pragma once

__global__ void demosaicPlanarKernel(const float* normalisedBayer, float* rgbPlanes,
                                     int imageWidth, int imageHeight);

__global__ void whiteBalancePlanarKernel(float* rgbPlanes, int imageWidth, int imageHeight,
                                         float redGain, float greenGain, float blueGain);

__global__ void denoisePlanarKernel(const float* rgbPlanes, float* denoisedPlanes,
                                    int imageWidth, int imageHeight);

// Reads planar floats, writes INTERLEAVED bytes, because the output file
// (and the CPU reference) store R G B together for each pixel.
__global__ void gammaPlanarKernel(const float* rgbPlanes, unsigned char* outputBytes,
                                  int imageWidth, int imageHeight);
