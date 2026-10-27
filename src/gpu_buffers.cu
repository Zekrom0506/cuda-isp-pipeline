// gpu_buffers.cu
// The cudaMalloc / cudaFree calls from Milestone 2, moved into functions.

#include "gpu_buffers.h"

#include "cuda_helpers.h"

void allocateDeviceBuffers(DeviceBuffers& buffers, int pixelCount) {
    size_t pixels = static_cast<size_t>(pixelCount);
    buffers.bayerBytes = pixels * sizeof(uint16_t);
    buffers.normalisedBytes = pixels * sizeof(float);
    buffers.rgbFloatBytes = pixels * 3 * sizeof(float);
    buffers.outputBytes = pixels * 3 * sizeof(unsigned char);

    checkCudaError(cudaMalloc(&buffers.deviceBayer, buffers.bayerBytes),
                   "allocating Bayer buffer");
    checkCudaError(cudaMalloc(&buffers.deviceNormalised, buffers.normalisedBytes),
                   "allocating normalised buffer");
    checkCudaError(cudaMalloc(&buffers.deviceRgb, buffers.rgbFloatBytes),
                   "allocating RGB buffer");
    checkCudaError(cudaMalloc(&buffers.deviceDenoised, buffers.rgbFloatBytes),
                   "allocating denoised buffer");
    checkCudaError(cudaMalloc(&buffers.deviceOutput, buffers.outputBytes),
                   "allocating output buffer");
}

void freeDeviceBuffers(DeviceBuffers& buffers) {
    checkCudaError(cudaFree(buffers.deviceBayer), "freeing Bayer buffer");
    checkCudaError(cudaFree(buffers.deviceNormalised), "freeing normalised buffer");
    checkCudaError(cudaFree(buffers.deviceRgb), "freeing RGB buffer");
    checkCudaError(cudaFree(buffers.deviceDenoised), "freeing denoised buffer");
    checkCudaError(cudaFree(buffers.deviceOutput), "freeing output buffer");
}
