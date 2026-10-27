// cuda_helpers.h
// Small helpers shared by every CUDA program in this project.

#pragma once

// cuda_runtime.h declares the CUDA functions we call from the CPU side:
// cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize and so on.
#include <cuda_runtime.h>

// Every CUDA API call returns a cudaError_t. cudaSuccess means it worked.
// CUDA does NOT stop the program by itself when something fails; it just
// returns an error code that is easy to ignore. Ignored errors show up later
// as wrong pictures or confusing crashes, so we check after every call.
// If the result is an error, this prints what we were doing and exits.
void checkCudaError(cudaError_t result, const char* whatWeWereDoing);

// Block size used for every 2D kernel launch: 16 x 16 = 256 threads per block.
// 256 is a common, safe choice: big enough to keep the GPU busy, well under
// the limit of 1024 threads per block.
const int BLOCK_WIDTH = 16;
const int BLOCK_HEIGHT = 16;
