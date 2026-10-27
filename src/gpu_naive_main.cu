// gpu_naive_main.cu
// Milestone 2: the naive GPU pipeline.
// Plan: copy the Bayer data to the GPU ONCE, run all 5 kernels one after
// another with every intermediate image staying in GPU memory, then copy the
// final 8-bit image back ONCE. Copies between CPU and GPU are slow, so we
// avoid copying anything in between stages.

#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_kernels.h"
#include "isp_io.h"

int main() {
    // ---- Load input on the CPU (same files as the CPU version) ----
    ImageMetadata metadata;
    bool metadataOk = readMetadataFile("data/metadata.txt", metadata);
    if (!metadataOk) {
        return 1;
    }
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;
    int pixelCount = imageWidth * imageHeight;
    std::cout << "Image: " << imageWidth << " x " << imageHeight << std::endl;

    std::vector<uint16_t> bayerPixels;
    bool bayerOk = readBayerFile("data/bayer.bin", pixelCount, bayerPixels);
    if (!bayerOk) {
        return 1;
    }

    // ---- Work out the size of each buffer in bytes ----
    // size_t is an unsigned integer type big enough for any memory size.
    size_t bayerBytes = static_cast<size_t>(pixelCount) * sizeof(uint16_t);
    size_t normalisedBytes = static_cast<size_t>(pixelCount) * sizeof(float);
    size_t rgbFloatBytes = static_cast<size_t>(pixelCount) * 3 * sizeof(float);
    size_t outputBytesCount = static_cast<size_t>(pixelCount) * 3 * sizeof(unsigned char);

    // ---- Allocate GPU memory ----
    // cudaMalloc reserves memory on the GPU (like malloc, but on the GPU) and
    // stores its address in our pointer. We pass the ADDRESS of the pointer
    // (&deviceBayer) so cudaMalloc can fill it in. We name GPU pointers
    // "device..." because CPU code must never read through them directly.
    uint16_t* deviceBayer = nullptr;
    float* deviceNormalised = nullptr;
    float* deviceRgb = nullptr;
    float* deviceDenoised = nullptr;
    unsigned char* deviceOutput = nullptr;

    checkCudaError(cudaMalloc(&deviceBayer, bayerBytes), "allocating Bayer buffer");
    checkCudaError(cudaMalloc(&deviceNormalised, normalisedBytes), "allocating normalised buffer");
    checkCudaError(cudaMalloc(&deviceRgb, rgbFloatBytes), "allocating RGB buffer");
    checkCudaError(cudaMalloc(&deviceDenoised, rgbFloatBytes), "allocating denoised buffer");
    checkCudaError(cudaMalloc(&deviceOutput, outputBytesCount), "allocating output buffer");

    // ---- Copy the input to the GPU (once) ----
    // cudaMemcpy(destination, source, byte count, direction).
    // cudaMemcpyHostToDevice means CPU memory ("host") to GPU memory ("device").
    checkCudaError(cudaMemcpy(deviceBayer, bayerPixels.data(), bayerBytes, cudaMemcpyHostToDevice),
                   "copying Bayer data to the GPU");

    // ---- Choose the launch shape ----
    // dim3 is a CUDA type holding up to 3 sizes (x, y, z); unset ones are 1.
    // blockShape: 16 x 16 threads per block.
    // gridShape: enough blocks to cover the image. The "+ BLOCK_WIDTH - 1"
    // rounds UP, so a partial block covers the leftover pixels at the edge.
    // Example: width 5796 / 16 = 362.25, so we need 363 blocks across.
    dim3 blockShape(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocksAcross = (imageWidth + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    int blocksDown = (imageHeight + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    dim3 gridShape(blocksAcross, blocksDown);
    std::cout << "Grid: " << blocksAcross << " x " << blocksDown << " blocks of "
              << BLOCK_WIDTH << " x " << BLOCK_HEIGHT << " threads" << std::endl;

    // ---- Run the 5 kernels ----
    // kernelName<<<gridShape, blockShape>>>(arguments) LAUNCHES a kernel: it
    // starts gridShape x blockShape threads, all running the kernel at once.
    // The launch returns to the CPU immediately, without waiting for the GPU.
    // Kernels launched one after another still run IN ORDER on the GPU, so
    // stage 2 never starts before stage 1 has finished.
    //
    // A launch itself returns nothing, so we ask cudaGetLastError() whether
    // the launch was accepted (for example, a bad block size is caught here).
    blackLevelKernel<<<gridShape, blockShape>>>(deviceBayer, deviceNormalised,
                                                imageWidth, imageHeight,
                                                metadata.blackLevel, metadata.whiteLevel);
    checkCudaError(cudaGetLastError(), "launching blackLevelKernel");

    demosaicKernel<<<gridShape, blockShape>>>(deviceNormalised, deviceRgb, imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching demosaicKernel");

    whiteBalanceKernel<<<gridShape, blockShape>>>(deviceRgb, imageWidth, imageHeight,
                                                  metadata.whiteBalanceRed,
                                                  metadata.whiteBalanceGreen,
                                                  metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching whiteBalanceKernel");

    denoiseKernel<<<gridShape, blockShape>>>(deviceRgb, deviceDenoised, imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching denoiseKernel");

    gammaKernel<<<gridShape, blockShape>>>(deviceDenoised, deviceOutput, imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching gammaKernel");

    // cudaDeviceSynchronize makes the CPU wait until the GPU has finished ALL
    // the work launched so far. Errors that happen WHILE a kernel runs (for
    // example reading outside a buffer) are only reported here.
    checkCudaError(cudaDeviceSynchronize(), "running the kernels");

    // ---- Copy the result back to the CPU (once) ----
    // cudaMemcpyDeviceToHost is the opposite direction: GPU to CPU.
    std::vector<unsigned char> outputBytes(outputBytesCount);
    checkCudaError(cudaMemcpy(outputBytes.data(), deviceOutput, outputBytesCount, cudaMemcpyDeviceToHost),
                   "copying the output back to the CPU");

    // ---- Free GPU memory ----
    // cudaFree gives GPU memory back, like free() does on the CPU.
    checkCudaError(cudaFree(deviceBayer), "freeing Bayer buffer");
    checkCudaError(cudaFree(deviceNormalised), "freeing normalised buffer");
    checkCudaError(cudaFree(deviceRgb), "freeing RGB buffer");
    checkCudaError(cudaFree(deviceDenoised), "freeing denoised buffer");
    checkCudaError(cudaFree(deviceOutput), "freeing output buffer");

    bool writeOk = writeRgbFile("output/gpu_naive_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    std::cout << "Saved output/gpu_naive_output.bin" << std::endl;
    return 0;
}
