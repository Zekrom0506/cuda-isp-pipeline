// gpu_kernels.h
// The five ISP stages as CUDA kernels (naive version: one thread per pixel).
// Same image layouts as the CPU version (see cpu_stages.h).
//
// "__global__" marks a KERNEL: a function that runs on the GPU but is started
// (launched) from CPU code. Thousands of copies run at the same time, one per
// thread. A kernel must return void; it writes its results into GPU memory.
//
// All pointers passed to these kernels point to GPU memory (from cudaMalloc).
// The GPU cannot read normal CPU memory such as a std::vector.

#pragma once

#include <cstdint>

// Stage 1 (gpu_pointwise_kernels.cu)
__global__ void blackLevelKernel(const uint16_t* bayerPixels, float* normalisedBayer,
                                 int imageWidth, int imageHeight,
                                 float blackLevel, float whiteLevel);

// Stage 2 (gpu_neighbourhood_kernels.cu)
__global__ void demosaicKernel(const float* normalisedBayer, float* rgbImage,
                               int imageWidth, int imageHeight);

// Stage 3 (gpu_pointwise_kernels.cu) - changes rgbImage in place.
__global__ void whiteBalanceKernel(float* rgbImage, int imageWidth, int imageHeight,
                                   float redGain, float greenGain, float blueGain);

// Stage 4 (gpu_neighbourhood_kernels.cu)
__global__ void denoiseKernel(const float* rgbImage, float* denoisedImage,
                              int imageWidth, int imageHeight);

// Stage 5 (gpu_pointwise_kernels.cu)
__global__ void gammaKernel(const float* rgbImage, unsigned char* outputBytes,
                            int imageWidth, int imageHeight);
