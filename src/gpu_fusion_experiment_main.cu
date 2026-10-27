// gpu_fusion_experiment_main.cu
// Milestone 7: compares the best 5-kernel pipeline (E: planar, 32x8 blocks)
// with the single fused kernel (G). Both are timed from "first kernel starts"
// to "last kernel ends", 1 warm-up + 10 averaged runs, measured E G E G.

#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_buffers.h"
#include "gpu_fused_kernel.h"
#include "gpu_kernels.h"
#include "gpu_planar_kernels.h"
#include "isp_io.h"

const int TIMED_RUN_COUNT = 10;

// Launches the five separate kernels (configuration E from Milestone 5).
static void launchFiveKernels(DeviceBuffers& buffers, const ImageMetadata& metadata,
                              dim3 gridShape, dim3 blockShape) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;

    blackLevelKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceNormalised,
                                                imageWidth, imageHeight,
                                                metadata.blackLevel, metadata.whiteLevel);
    checkCudaError(cudaGetLastError(), "launching blackLevelKernel");
    demosaicPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                                    imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching demosaicPlanarKernel");
    whiteBalancePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                        metadata.whiteBalanceRed,
                                                        metadata.whiteBalanceGreen,
                                                        metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching whiteBalancePlanarKernel");
    denoisePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                   imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching denoisePlanarKernel");
    gammaPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                                 imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching gammaPlanarKernel");
}

// Launches the single fused kernel. It reads the raw Bayer data and writes the
// final bytes; the in-between buffers are not used at all.
static void launchFusedKernel(DeviceBuffers& buffers, const ImageMetadata& metadata,
                              dim3 gridShape, dim3 blockShape) {
    fusedPipelineKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceOutput,
                                                   metadata.imageWidth, metadata.imageHeight,
                                                   metadata.blackLevel, metadata.whiteLevel,
                                                   metadata.whiteBalanceRed,
                                                   metadata.whiteBalanceGreen,
                                                   metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching fusedPipelineKernel");
}

// Times one version: 1 warm-up run (ignored), then the average of 10 runs.
static float averageMilliseconds(bool useFusedKernel, DeviceBuffers& buffers,
                                 const ImageMetadata& metadata,
                                 dim3 gridShape, dim3 blockShape,
                                 cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    float totalMilliseconds = 0.0f;

    for (int runNumber = 0; runNumber <= TIMED_RUN_COUNT; runNumber++) {
        checkCudaError(cudaEventRecord(startEvent), "recording start event");
        if (useFusedKernel) {
            launchFusedKernel(buffers, metadata, gridShape, blockShape);
        } else {
            launchFiveKernels(buffers, metadata, gridShape, blockShape);
        }
        checkCudaError(cudaEventRecord(stopEvent), "recording stop event");
        checkCudaError(cudaEventSynchronize(stopEvent), "waiting for the run");

        float runMilliseconds = 0.0f;
        checkCudaError(cudaEventElapsedTime(&runMilliseconds, startEvent, stopEvent),
                       "reading run time");

        // Run 0 is the warm-up: measured, but not added to the average.
        if (runNumber > 0) {
            totalMilliseconds = totalMilliseconds + runMilliseconds;
        }
    }
    return totalMilliseconds / TIMED_RUN_COUNT;
}

int main() {
    ImageMetadata metadata;
    bool metadataOk = readMetadataFile("data/metadata.txt", metadata);
    if (!metadataOk) {
        return 1;
    }
    int pixelCount = metadata.imageWidth * metadata.imageHeight;
    std::vector<uint16_t> bayerPixels;
    bool bayerOk = readBayerFile("data/bayer.bin", pixelCount, bayerPixels);
    if (!bayerOk) {
        return 1;
    }

    DeviceBuffers buffers;
    allocateDeviceBuffers(buffers, pixelCount);
    checkCudaError(cudaMemcpy(buffers.deviceBayer, bayerPixels.data(), buffers.bayerBytes,
                              cudaMemcpyHostToDevice), "copying Bayer data to the GPU");

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    checkCudaError(cudaEventCreate(&startEvent), "creating start event");
    checkCudaError(cudaEventCreate(&stopEvent), "creating stop event");

    // Both versions use 32 x 8 blocks, so they can share one launch shape.
    dim3 blockShape(FUSED_TILE_WIDTH, FUSED_TILE_HEIGHT);
    int blocksAcross = (metadata.imageWidth + FUSED_TILE_WIDTH - 1) / FUSED_TILE_WIDTH;
    int blocksDown = (metadata.imageHeight + FUSED_TILE_HEIGHT - 1) / FUSED_TILE_HEIGHT;
    dim3 gridShape(blocksAcross, blocksDown);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time for all ISP stages, " << TIMED_RUN_COUNT << " warm runs each:" << std::endl;

    float fiveKernelFirst = averageMilliseconds(false, buffers, metadata, gridShape, blockShape, startEvent, stopEvent);
    std::cout << "  E  5 separate kernels : " << fiveKernelFirst << " ms" << std::endl;
    float fusedFirst = averageMilliseconds(true, buffers, metadata, gridShape, blockShape, startEvent, stopEvent);
    std::cout << "  G  1 fused kernel     : " << fusedFirst << " ms" << std::endl;
    float fiveKernelSecond = averageMilliseconds(false, buffers, metadata, gridShape, blockShape, startEvent, stopEvent);
    std::cout << "  E  again              : " << fiveKernelSecond << " ms" << std::endl;
    float fusedSecond = averageMilliseconds(true, buffers, metadata, gridShape, blockShape, startEvent, stopEvent);
    std::cout << "  G  again              : " << fusedSecond << " ms" << std::endl;

    float fiveKernelAverage = (fiveKernelFirst + fiveKernelSecond) / 2.0f;
    float fusedAverage = (fusedFirst + fusedSecond) / 2.0f;
    std::cout << "Speed-up of fused over 5 kernels: " << fiveKernelAverage / fusedAverage << "x" << std::endl;

    // GPU memory each version needs. The fused kernel only needs input and output.
    double fiveKernelMegabytes = (buffers.bayerBytes + buffers.normalisedBytes
                                  + 2 * buffers.rgbFloatBytes + buffers.outputBytes) / 1000000.0;
    double fusedMegabytes = (buffers.bayerBytes + buffers.outputBytes) / 1000000.0;
    std::cout << "GPU memory needed: 5 kernels " << fiveKernelMegabytes
              << " MB, fused " << fusedMegabytes << " MB" << std::endl;

    // The last run was the fused kernel, so deviceOutput holds its result.
    std::vector<unsigned char> outputBytes(buffers.outputBytes);
    checkCudaError(cudaMemcpy(outputBytes.data(), buffers.deviceOutput, buffers.outputBytes,
                              cudaMemcpyDeviceToHost), "copying output to the CPU");

    checkCudaError(cudaEventDestroy(startEvent), "destroying start event");
    checkCudaError(cudaEventDestroy(stopEvent), "destroying stop event");
    freeDeviceBuffers(buffers);

    bool writeOk = writeRgbFile("output/gpu_fused_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    std::cout << "Saved output/gpu_fused_output.bin" << std::endl;
    return 0;
}
