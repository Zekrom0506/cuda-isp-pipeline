// gpu_shared_experiment_main.cu
// Milestone 6: compares the planar 32x8 pipeline (best from Milestone 5) with
// the same pipeline using the shared-memory denoise kernel.
// It also checks that both denoise kernels give exactly the same floats.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_buffers.h"
#include "gpu_kernels.h"
#include "gpu_planar_kernels.h"
#include "gpu_shared_denoise.h"
#include "isp_io.h"

const int STAGE_COUNT = 5;
const int TIMED_RUN_COUNT = 10;

static void runPipelineOnce(DeviceBuffers& buffers, float* deviceDenoisedTarget,
                            const ImageMetadata& metadata, bool useSharedDenoise,
                            dim3 gridShape, dim3 blockShape,
                            cudaEvent_t stageEvents[], float stageMilliseconds[]) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;

    checkCudaError(cudaEventRecord(stageEvents[0]), "recording event 0");

    blackLevelKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceNormalised,
                                                imageWidth, imageHeight,
                                                metadata.blackLevel, metadata.whiteLevel);
    checkCudaError(cudaGetLastError(), "launching blackLevelKernel");
    checkCudaError(cudaEventRecord(stageEvents[1]), "recording event 1");

    demosaicPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                                    imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching demosaicPlanarKernel");
    checkCudaError(cudaEventRecord(stageEvents[2]), "recording event 2");

    whiteBalancePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                        metadata.whiteBalanceRed,
                                                        metadata.whiteBalanceGreen,
                                                        metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching whiteBalancePlanarKernel");
    checkCudaError(cudaEventRecord(stageEvents[3]), "recording event 3");

    if (useSharedDenoise) {
        denoiseSharedKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, deviceDenoisedTarget,
                                                       imageWidth, imageHeight);
    } else {
        denoisePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, deviceDenoisedTarget,
                                                       imageWidth, imageHeight);
    }
    checkCudaError(cudaGetLastError(), "launching denoise");
    checkCudaError(cudaEventRecord(stageEvents[4]), "recording event 4");

    gammaPlanarKernel<<<gridShape, blockShape>>>(deviceDenoisedTarget, buffers.deviceOutput,
                                                 imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching gammaPlanarKernel");
    checkCudaError(cudaEventRecord(stageEvents[5]), "recording event 5");

    checkCudaError(cudaEventSynchronize(stageEvents[STAGE_COUNT]), "waiting for the pipeline");
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        checkCudaError(cudaEventElapsedTime(&stageMilliseconds[stage],
                                            stageEvents[stage], stageEvents[stage + 1]),
                       "reading stage time");
    }
}

// 1 warm-up run, then TIMED_RUN_COUNT runs, averaged per stage, printed as one row.
static void measureAndPrint(const char* rowName, DeviceBuffers& buffers, float* deviceDenoisedTarget,
                            const ImageMetadata& metadata, bool useSharedDenoise,
                            dim3 gridShape, dim3 blockShape, cudaEvent_t stageEvents[]) {
    float thisRunMilliseconds[STAGE_COUNT];
    float averageMilliseconds[STAGE_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    runPipelineOnce(buffers, deviceDenoisedTarget, metadata, useSharedDenoise,
                    gridShape, blockShape, stageEvents, thisRunMilliseconds);

    for (int runNumber = 0; runNumber < TIMED_RUN_COUNT; runNumber++) {
        runPipelineOnce(buffers, deviceDenoisedTarget, metadata, useSharedDenoise,
                        gridShape, blockShape, stageEvents, thisRunMilliseconds);
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            averageMilliseconds[stage] = averageMilliseconds[stage] + thisRunMilliseconds[stage];
        }
    }

    float totalMilliseconds = 0.0f;
    std::cout << std::setw(24) << rowName;
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        averageMilliseconds[stage] = averageMilliseconds[stage] / TIMED_RUN_COUNT;
        totalMilliseconds = totalMilliseconds + averageMilliseconds[stage];
        std::cout << std::setw(10) << averageMilliseconds[stage];
    }
    std::cout << std::setw(10) << totalMilliseconds << std::endl;
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

    // A second denoise output buffer, so we can keep both results and compare them.
    float* deviceDenoisedShared = nullptr;
    checkCudaError(cudaMalloc(&deviceDenoisedShared, buffers.rgbFloatBytes),
                   "allocating second denoise buffer");

    cudaEvent_t stageEvents[STAGE_COUNT + 1];
    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventCreate(&stageEvents[eventNumber]), "creating stage event");
    }

    dim3 blockShape(TILE_WIDTH, TILE_HEIGHT);
    int blocksAcross = (metadata.imageWidth + TILE_WIDTH - 1) / TILE_WIDTH;
    int blocksDown = (metadata.imageHeight + TILE_HEIGHT - 1) / TILE_HEIGHT;
    dim3 gridShape(blocksAcross, blocksDown);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average kernel times in ms (" << TIMED_RUN_COUNT << " warm runs each)" << std::endl;
    std::cout << std::setw(24) << "version"
              << std::setw(10) << "black" << std::setw(10) << "demosaic"
              << std::setw(10) << "white b." << std::setw(10) << "denoise"
              << std::setw(10) << "gamma" << std::setw(10) << "TOTAL" << std::endl;

    // Measure each version twice, alternating, so a slow drift of the GPU clock
    // during the program cannot unfairly favour whichever version runs first.
    measureAndPrint("E planar 32x8 (global)", buffers, buffers.deviceDenoised, metadata, false,
                    gridShape, blockShape, stageEvents);
    measureAndPrint("F shared-memory denoise", buffers, deviceDenoisedShared, metadata, true,
                    gridShape, blockShape, stageEvents);
    measureAndPrint("E again", buffers, buffers.deviceDenoised, metadata, false,
                    gridShape, blockShape, stageEvents);
    measureAndPrint("F again", buffers, deviceDenoisedShared, metadata, true,
                    gridShape, blockShape, stageEvents);

    // ---- Compare the two denoise results float by float ----
    std::vector<float> denoisedGlobal(pixelCount * 3);
    std::vector<float> denoisedShared(pixelCount * 3);
    checkCudaError(cudaMemcpy(denoisedGlobal.data(), buffers.deviceDenoised, buffers.rgbFloatBytes,
                              cudaMemcpyDeviceToHost), "copying global-memory denoise result");
    checkCudaError(cudaMemcpy(denoisedShared.data(), deviceDenoisedShared, buffers.rgbFloatBytes,
                              cudaMemcpyDeviceToHost), "copying shared-memory denoise result");

    float largestFloatDifference = 0.0f;
    for (int valueIndex = 0; valueIndex < pixelCount * 3; valueIndex++) {
        float difference = std::fabs(denoisedGlobal[valueIndex] - denoisedShared[valueIndex]);
        if (difference > largestFloatDifference) {
            largestFloatDifference = difference;
        }
    }
    std::cout << std::setprecision(9);
    std::cout << "Largest float difference, global vs shared denoise: "
              << largestFloatDifference << std::endl;

    // The last run was F, so deviceOutput holds the shared-memory pipeline result.
    std::vector<unsigned char> outputBytes(buffers.outputBytes);
    checkCudaError(cudaMemcpy(outputBytes.data(), buffers.deviceOutput, buffers.outputBytes,
                              cudaMemcpyDeviceToHost), "copying output to the CPU");

    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventDestroy(stageEvents[eventNumber]), "destroying stage event");
    }
    checkCudaError(cudaFree(deviceDenoisedShared), "freeing second denoise buffer");
    freeDeviceBuffers(buffers);

    bool writeOk = writeRgbFile("output/gpu_shared_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    std::cout << "Saved output/gpu_shared_output.bin" << std::endl;
    return 0;
}
