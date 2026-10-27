// gpu_naive_timed_main.cu
// Milestone 4: the same naive pipeline as Milestone 2, now with careful timing.
//   - Each kernel is timed separately with CUDA events.
//   - The copies between CPU and GPU are timed separately from the kernels.
//   - Run 1 (the "cold" warm-up run) is reported on its own.
//   - Runs 2 to 11 are averaged to get a stable number.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_buffers.h"
#include "gpu_kernels.h"
#include "isp_io.h"

const int STAGE_COUNT = 5;
const int TIMED_RUN_COUNT = 10;
const char* STAGE_NAMES[STAGE_COUNT] = {
    "Stage 1 black level",
    "Stage 2 demosaic   ",
    "Stage 3 white bal. ",
    "Stage 4 denoise    ",
    "Stage 5 gamma      "
};

// ---------------------------------------------------------------------------
// Why CUDA events instead of std::chrono?
// A kernel launch returns to the CPU straight away, before the GPU has done
// the work. A CPU clock around a launch measures only "how long it took to
// ASK", not how long the GPU worked. A CUDA event is a marker we drop into the
// GPU's queue of work. The GPU writes a timestamp when it reaches the marker,
// so the gap between two events is real GPU time.
//
//   cudaEventCreate(&event)        - make an event object
//   cudaEventRecord(event)         - put the marker into the GPU's work queue
//   cudaEventSynchronize(event)    - make the CPU wait until the GPU reaches it
//   cudaEventElapsedTime(&ms, a, b)- milliseconds between markers a and b
//   cudaEventDestroy(event)        - free the event object
// ---------------------------------------------------------------------------

// Runs all 5 kernels once. stageEvents must hold STAGE_COUNT + 1 events:
// event 0 before kernel 1, event 1 after kernel 1 (= before kernel 2), and so on.
static void runPipelineOnce(DeviceBuffers& buffers, const ImageMetadata& metadata,
                            dim3 gridShape, dim3 blockShape,
                            cudaEvent_t stageEvents[], float stageMilliseconds[]) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;

    checkCudaError(cudaEventRecord(stageEvents[0]), "recording event before stage 1");

    blackLevelKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceNormalised,
                                                imageWidth, imageHeight,
                                                metadata.blackLevel, metadata.whiteLevel);
    checkCudaError(cudaGetLastError(), "launching blackLevelKernel");
    checkCudaError(cudaEventRecord(stageEvents[1]), "recording event after stage 1");

    demosaicKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                              imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching demosaicKernel");
    checkCudaError(cudaEventRecord(stageEvents[2]), "recording event after stage 2");

    whiteBalanceKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                  metadata.whiteBalanceRed,
                                                  metadata.whiteBalanceGreen,
                                                  metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching whiteBalanceKernel");
    checkCudaError(cudaEventRecord(stageEvents[3]), "recording event after stage 3");

    denoiseKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                             imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching denoiseKernel");
    checkCudaError(cudaEventRecord(stageEvents[4]), "recording event after stage 4");

    gammaKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                           imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching gammaKernel");
    checkCudaError(cudaEventRecord(stageEvents[5]), "recording event after stage 5");

    // Wait for the last marker, so every timestamp exists before we read them.
    checkCudaError(cudaEventSynchronize(stageEvents[STAGE_COUNT]), "waiting for the pipeline");

    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        checkCudaError(cudaEventElapsedTime(&stageMilliseconds[stage],
                                            stageEvents[stage], stageEvents[stage + 1]),
                       "reading stage time");
    }
}

static void printStageTimes(const char* heading, const float stageMilliseconds[]) {
    std::cout << heading << std::endl;
    float totalMilliseconds = 0.0f;
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        std::cout << "  " << STAGE_NAMES[stage] << " : " << stageMilliseconds[stage] << " ms" << std::endl;
        totalMilliseconds = totalMilliseconds + stageMilliseconds[stage];
    }
    std::cout << "  All 5 kernels       : " << totalMilliseconds << " ms" << std::endl;
}

int main() {
    // Print every number with 3 digits after the decimal point.
    std::cout << std::fixed << std::setprecision(3);

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
    std::cout << "Image: " << metadata.imageWidth << " x " << metadata.imageHeight << std::endl;

    // ---- CUDA start-up cost ----
    // The very first CUDA call creates a "context" (the GPU-side setup for this
    // program). That is slow and happens only once. cudaFree(0) frees nothing,
    // but forces the start-up to happen here, so it does not pollute other timings.
    // We use std::chrono for this one because no GPU work is queued yet.
    std::chrono::steady_clock::time_point startupBegin = std::chrono::steady_clock::now();
    checkCudaError(cudaFree(0), "starting CUDA");
    std::chrono::steady_clock::time_point startupEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> startupTime = startupEnd - startupBegin;
    std::cout << "CUDA start-up (once per program): " << startupTime.count() << " ms" << std::endl;

    DeviceBuffers buffers;
    allocateDeviceBuffers(buffers, pixelCount);

    cudaEvent_t copyStart;
    cudaEvent_t copyStop;
    checkCudaError(cudaEventCreate(&copyStart), "creating copyStart event");
    checkCudaError(cudaEventCreate(&copyStop), "creating copyStop event");

    cudaEvent_t stageEvents[STAGE_COUNT + 1];
    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventCreate(&stageEvents[eventNumber]), "creating stage event");
    }

    // ---- Time the copy CPU -> GPU ----
    checkCudaError(cudaEventRecord(copyStart), "recording copyStart");
    checkCudaError(cudaMemcpy(buffers.deviceBayer, bayerPixels.data(), buffers.bayerBytes,
                              cudaMemcpyHostToDevice), "copying Bayer data to the GPU");
    checkCudaError(cudaEventRecord(copyStop), "recording copyStop");
    checkCudaError(cudaEventSynchronize(copyStop), "waiting for copy to GPU");
    float copyToGpuMilliseconds = 0.0f;
    checkCudaError(cudaEventElapsedTime(&copyToGpuMilliseconds, copyStart, copyStop),
                   "reading copy-to-GPU time");

    dim3 blockShape(BLOCK_WIDTH, BLOCK_HEIGHT);
    int blocksAcross = (metadata.imageWidth + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    int blocksDown = (metadata.imageHeight + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    dim3 gridShape(blocksAcross, blocksDown);

    // ---- Run 1: the cold (warm-up) run, reported on its own ----
    float coldStageMilliseconds[STAGE_COUNT];
    runPipelineOnce(buffers, metadata, gridShape, blockShape, stageEvents, coldStageMilliseconds);
    printStageTimes("Run 1 (cold, includes one-time warm-up):", coldStageMilliseconds);

    // ---- Runs 2 to 11: add up each stage, then divide to get the average ----
    float summedStageMilliseconds[STAGE_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int runNumber = 0; runNumber < TIMED_RUN_COUNT; runNumber++) {
        float thisRunMilliseconds[STAGE_COUNT];
        runPipelineOnce(buffers, metadata, gridShape, blockShape, stageEvents, thisRunMilliseconds);
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            summedStageMilliseconds[stage] = summedStageMilliseconds[stage] + thisRunMilliseconds[stage];
        }
    }
    float averageStageMilliseconds[STAGE_COUNT];
    float averageKernelTotal = 0.0f;
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        averageStageMilliseconds[stage] = summedStageMilliseconds[stage] / TIMED_RUN_COUNT;
        averageKernelTotal = averageKernelTotal + averageStageMilliseconds[stage];
    }
    printStageTimes("Average of runs 2 to 11 (warm):", averageStageMilliseconds);

    // ---- Time the copy GPU -> CPU ----
    std::vector<unsigned char> outputBytes(buffers.outputBytes);
    checkCudaError(cudaEventRecord(copyStart), "recording copyStart");
    checkCudaError(cudaMemcpy(outputBytes.data(), buffers.deviceOutput, buffers.outputBytes,
                              cudaMemcpyDeviceToHost), "copying output to the CPU");
    checkCudaError(cudaEventRecord(copyStop), "recording copyStop");
    checkCudaError(cudaEventSynchronize(copyStop), "waiting for copy to CPU");
    float copyToCpuMilliseconds = 0.0f;
    checkCudaError(cudaEventElapsedTime(&copyToCpuMilliseconds, copyStart, copyStop),
                   "reading copy-to-CPU time");

    float totalWithCopies = copyToGpuMilliseconds + averageKernelTotal + copyToCpuMilliseconds;
    std::cout << "Copy CPU -> GPU (" << buffers.bayerBytes / 1000000.0 << " MB) : "
              << copyToGpuMilliseconds << " ms" << std::endl;
    std::cout << "Copy GPU -> CPU (" << buffers.outputBytes / 1000000.0 << " MB) : "
              << copyToCpuMilliseconds << " ms" << std::endl;
    std::cout << "Warm total = copy in + kernels + copy out : " << totalWithCopies << " ms" << std::endl;

    // ---- Clean up ----
    checkCudaError(cudaEventDestroy(copyStart), "destroying copyStart");
    checkCudaError(cudaEventDestroy(copyStop), "destroying copyStop");
    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventDestroy(stageEvents[eventNumber]), "destroying stage event");
    }
    freeDeviceBuffers(buffers);

    bool writeOk = writeRgbFile("output/gpu_naive_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    std::cout << "Saved output/gpu_naive_output.bin" << std::endl;
    return 0;
}
