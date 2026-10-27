// gpu_coalescing_experiment_main.cu
// Milestone 5: measures how memory access patterns change kernel speed.
// Five configurations, each timed the Milestone 4 way (1 warm-up + 10 averaged):
//   A. interleaved RGB,  1 x 32 blocks  - deliberately BAD (uncoalesced demo)
//   B. interleaved RGB, 16 x 16 blocks  - Milestone 4, our starting point
//   C. interleaved RGB, 32 x  8 blocks  - a warp now covers one full row segment
//   D. planar RGB,      16 x 16 blocks  - colours stored in separate planes
//   E. planar RGB,      32 x  8 blocks  - both fixes together
// Every configuration uses the same 256 threads per block except A (32), so
// the comparison is mostly about WHICH addresses a warp touches together.

#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_buffers.h"
#include "gpu_kernels.h"
#include "gpu_planar_kernels.h"
#include "isp_io.h"

const int STAGE_COUNT = 5;
const int TIMED_RUN_COUNT = 10;
const int CONFIGURATION_COUNT = 5;

// Runs the 5 stages once, using either the interleaved kernels (Milestone 2)
// or the planar kernels (this milestone). Fills stageMilliseconds[0..4].
static void runPipelineOnce(DeviceBuffers& buffers, const ImageMetadata& metadata,
                            bool usePlanarLayout, dim3 gridShape, dim3 blockShape,
                            cudaEvent_t stageEvents[], float stageMilliseconds[]) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;

    checkCudaError(cudaEventRecord(stageEvents[0]), "recording event 0");

    // Stage 1 is identical for both layouts.
    blackLevelKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceNormalised,
                                                imageWidth, imageHeight,
                                                metadata.blackLevel, metadata.whiteLevel);
    checkCudaError(cudaGetLastError(), "launching blackLevelKernel");
    checkCudaError(cudaEventRecord(stageEvents[1]), "recording event 1");

    if (usePlanarLayout) {
        demosaicPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                                        imageWidth, imageHeight);
    } else {
        demosaicKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                                  imageWidth, imageHeight);
    }
    checkCudaError(cudaGetLastError(), "launching demosaic");
    checkCudaError(cudaEventRecord(stageEvents[2]), "recording event 2");

    if (usePlanarLayout) {
        whiteBalancePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                            metadata.whiteBalanceRed,
                                                            metadata.whiteBalanceGreen,
                                                            metadata.whiteBalanceBlue);
    } else {
        whiteBalanceKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                      metadata.whiteBalanceRed,
                                                      metadata.whiteBalanceGreen,
                                                      metadata.whiteBalanceBlue);
    }
    checkCudaError(cudaGetLastError(), "launching white balance");
    checkCudaError(cudaEventRecord(stageEvents[3]), "recording event 3");

    if (usePlanarLayout) {
        denoisePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                       imageWidth, imageHeight);
    } else {
        denoiseKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                 imageWidth, imageHeight);
    }
    checkCudaError(cudaGetLastError(), "launching denoise");
    checkCudaError(cudaEventRecord(stageEvents[4]), "recording event 4");

    if (usePlanarLayout) {
        gammaPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                                     imageWidth, imageHeight);
    } else {
        gammaKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                               imageWidth, imageHeight);
    }
    checkCudaError(cudaGetLastError(), "launching gamma");
    checkCudaError(cudaEventRecord(stageEvents[5]), "recording event 5");

    checkCudaError(cudaEventSynchronize(stageEvents[STAGE_COUNT]), "waiting for the pipeline");
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        checkCudaError(cudaEventElapsedTime(&stageMilliseconds[stage],
                                            stageEvents[stage], stageEvents[stage + 1]),
                       "reading stage time");
    }
}

// One warm-up run (thrown away), then TIMED_RUN_COUNT runs averaged per stage.
static void measureConfiguration(DeviceBuffers& buffers, const ImageMetadata& metadata,
                                 bool usePlanarLayout, int blockWidth, int blockHeight,
                                 cudaEvent_t stageEvents[], float averageMilliseconds[]) {
    dim3 blockShape(blockWidth, blockHeight);
    int blocksAcross = (metadata.imageWidth + blockWidth - 1) / blockWidth;
    int blocksDown = (metadata.imageHeight + blockHeight - 1) / blockHeight;
    dim3 gridShape(blocksAcross, blocksDown);

    float thisRunMilliseconds[STAGE_COUNT];
    runPipelineOnce(buffers, metadata, usePlanarLayout, gridShape, blockShape,
                    stageEvents, thisRunMilliseconds);

    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        averageMilliseconds[stage] = 0.0f;
    }
    for (int runNumber = 0; runNumber < TIMED_RUN_COUNT; runNumber++) {
        runPipelineOnce(buffers, metadata, usePlanarLayout, gridShape, blockShape,
                        stageEvents, thisRunMilliseconds);
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            averageMilliseconds[stage] = averageMilliseconds[stage] + thisRunMilliseconds[stage];
        }
    }
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        averageMilliseconds[stage] = averageMilliseconds[stage] / TIMED_RUN_COUNT;
    }
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

    cudaEvent_t stageEvents[STAGE_COUNT + 1];
    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventCreate(&stageEvents[eventNumber]), "creating stage event");
    }

    const char* configurationNames[CONFIGURATION_COUNT] = {
        "A interleaved  1x32",
        "B interleaved 16x16",
        "C interleaved  32x8",
        "D planar      16x16",
        "E planar       32x8"
    };
    bool configurationIsPlanar[CONFIGURATION_COUNT] = {false, false, false, true, true};
    int configurationBlockWidth[CONFIGURATION_COUNT] = {1, 16, 32, 16, 32};
    int configurationBlockHeight[CONFIGURATION_COUNT] = {32, 16, 8, 16, 8};

    // Table header. std::setw(n) pads the next printed item to n characters,
    // which lines the columns up.
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average kernel times in ms (" << TIMED_RUN_COUNT << " warm runs each)" << std::endl;
    std::cout << std::setw(21) << "configuration"
              << std::setw(10) << "black" << std::setw(10) << "demosaic"
              << std::setw(10) << "white b." << std::setw(10) << "denoise"
              << std::setw(10) << "gamma" << std::setw(10) << "TOTAL" << std::endl;

    for (int configuration = 0; configuration < CONFIGURATION_COUNT; configuration++) {
        float averageMilliseconds[STAGE_COUNT];
        measureConfiguration(buffers, metadata,
                             configurationIsPlanar[configuration],
                             configurationBlockWidth[configuration],
                             configurationBlockHeight[configuration],
                             stageEvents, averageMilliseconds);

        float totalMilliseconds = 0.0f;
        std::cout << std::setw(21) << configurationNames[configuration];
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            std::cout << std::setw(10) << averageMilliseconds[stage];
            totalMilliseconds = totalMilliseconds + averageMilliseconds[stage];
        }
        std::cout << std::setw(10) << totalMilliseconds << std::endl;
    }

    // The last configuration run was E (planar 32x8), so the output buffer now
    // holds its result. Save it so we can check it against the CPU.
    std::vector<unsigned char> outputBytes(buffers.outputBytes);
    checkCudaError(cudaMemcpy(outputBytes.data(), buffers.deviceOutput, buffers.outputBytes,
                              cudaMemcpyDeviceToHost), "copying output to the CPU");

    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventDestroy(stageEvents[eventNumber]), "destroying stage event");
    }
    freeDeviceBuffers(buffers);

    bool writeOk = writeRgbFile("output/gpu_planar_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    std::cout << "Saved output/gpu_planar_output.bin (configuration E)" << std::endl;
    return 0;
}
