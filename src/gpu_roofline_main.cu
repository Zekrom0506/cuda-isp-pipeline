// gpu_roofline_main.cu
// Milestone 9: collects the times the roofline analysis needs, all in ONE run
// so they are directly comparable:
//   1. the best memory bandwidth a simple copy kernel achieves on this GPU
//   2. per-stage times of the naive pipeline (interleaved, 16x16)
//   3. per-stage times of the planar pipeline (32x8)
//   4. the shared-memory denoise kernel on its own
//   5. the fused kernel
// Output lines look like "RESULT,<name>,<value>" for the Python script.
// Each measurement: 1 warm-up + 10 averaged runs (Milestone 4 method).

#include <iostream>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_buffers.h"
#include "gpu_fused_kernel.h"
#include "gpu_kernels.h"
#include "gpu_planar_kernels.h"
#include "gpu_shared_denoise.h"
#include "isp_io.h"

const int TIMED_RUN_COUNT = 10;
const int STAGE_COUNT = 5;
const char* STAGE_NAMES[STAGE_COUNT] = {"black", "demosaic", "white", "denoise", "gamma"};

// The simplest possible memory-bound kernel: each thread copies one float.
// One read + one write per thread, perfectly coalesced, almost no maths.
// Its speed is the practical memory-bandwidth ceiling for our kernels.
__global__ void copyKernel(const float* source, float* destination, int valueCount) {
    int valueIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (valueIndex >= valueCount) {
        return;
    }
    destination[valueIndex] = source[valueIndex];
}

static dim3 makeGridShape(const ImageMetadata& metadata, int blockWidth, int blockHeight) {
    int blocksAcross = (metadata.imageWidth + blockWidth - 1) / blockWidth;
    int blocksDown = (metadata.imageHeight + blockHeight - 1) / blockHeight;
    dim3 gridShape(blocksAcross, blocksDown);
    return gridShape;
}

static float elapsedMilliseconds(cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    float milliseconds = 0.0f;
    checkCudaError(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent), "reading elapsed time");
    return milliseconds;
}

static void measureCopyBandwidth(DeviceBuffers& buffers, cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    // Copy the big RGB float buffer (about 270 MB) into the other one.
    int valueCount = static_cast<int>(buffers.rgbFloatBytes / sizeof(float));
    int threadsPerBlock = 256;
    int blockCount = (valueCount + threadsPerBlock - 1) / threadsPerBlock;

    float totalMilliseconds = 0.0f;
    for (int runNumber = 0; runNumber <= TIMED_RUN_COUNT; runNumber++) {
        checkCudaError(cudaEventRecord(startEvent), "recording start event");
        copyKernel<<<blockCount, threadsPerBlock>>>(buffers.deviceRgb, buffers.deviceDenoised, valueCount);
        checkCudaError(cudaGetLastError(), "launching copyKernel");
        checkCudaError(cudaEventRecord(stopEvent), "recording stop event");
        checkCudaError(cudaEventSynchronize(stopEvent), "waiting for copyKernel");
        if (runNumber > 0) {
            totalMilliseconds = totalMilliseconds + elapsedMilliseconds(startEvent, stopEvent);
        }
    }
    float averageMilliseconds = totalMilliseconds / TIMED_RUN_COUNT;

    // Bytes moved = read the buffer once + write it once.
    double bytesMoved = 2.0 * static_cast<double>(buffers.rgbFloatBytes);
    double gigabytesPerSecond = bytesMoved / (averageMilliseconds / 1000.0) / 1.0e9;
    std::cout << "RESULT,copy_kernel_ms," << averageMilliseconds << std::endl;
    std::cout << "RESULT,copy_kernel_GBps," << gigabytesPerSecond << std::endl;
}

// Launches one stage of the naive (interleaved) or planar pipeline.
static void launchStage(int stage, bool usePlanarLayout, DeviceBuffers& buffers,
                        const ImageMetadata& metadata, dim3 gridShape, dim3 blockShape) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;

    if (stage == 0) {
        blackLevelKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceNormalised,
                                                    imageWidth, imageHeight,
                                                    metadata.blackLevel, metadata.whiteLevel);
    } else if (stage == 1 && usePlanarLayout) {
        demosaicPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                                        imageWidth, imageHeight);
    } else if (stage == 1) {
        demosaicKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                                  imageWidth, imageHeight);
    } else if (stage == 2 && usePlanarLayout) {
        whiteBalancePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                            metadata.whiteBalanceRed,
                                                            metadata.whiteBalanceGreen,
                                                            metadata.whiteBalanceBlue);
    } else if (stage == 2) {
        whiteBalanceKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                      metadata.whiteBalanceRed,
                                                      metadata.whiteBalanceGreen,
                                                      metadata.whiteBalanceBlue);
    } else if (stage == 3 && usePlanarLayout) {
        denoisePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                       imageWidth, imageHeight);
    } else if (stage == 3) {
        denoiseKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                 imageWidth, imageHeight);
    } else if (usePlanarLayout) {
        gammaPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                                     imageWidth, imageHeight);
    } else {
        gammaKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                               imageWidth, imageHeight);
    }
    checkCudaError(cudaGetLastError(), "launching a pipeline stage");
}

// Times each of the 5 stages separately and prints RESULT,<prefix>_<stage>,<ms>.
static void measureStages(const char* prefix, bool usePlanarLayout, int blockWidth, int blockHeight,
                          DeviceBuffers& buffers, const ImageMetadata& metadata,
                          cudaEvent_t stageEvents[]) {
    dim3 blockShape(blockWidth, blockHeight);
    dim3 gridShape = makeGridShape(metadata, blockWidth, blockHeight);
    float summedMilliseconds[STAGE_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int runNumber = 0; runNumber <= TIMED_RUN_COUNT; runNumber++) {
        checkCudaError(cudaEventRecord(stageEvents[0]), "recording event 0");
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            launchStage(stage, usePlanarLayout, buffers, metadata, gridShape, blockShape);
            checkCudaError(cudaEventRecord(stageEvents[stage + 1]), "recording stage event");
        }
        checkCudaError(cudaEventSynchronize(stageEvents[STAGE_COUNT]), "waiting for the pipeline");
        if (runNumber > 0) {
            for (int stage = 0; stage < STAGE_COUNT; stage++) {
                float stageMilliseconds = elapsedMilliseconds(stageEvents[stage], stageEvents[stage + 1]);
                summedMilliseconds[stage] = summedMilliseconds[stage] + stageMilliseconds;
            }
        }
    }
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        std::cout << "RESULT," << prefix << "_" << STAGE_NAMES[stage] << ","
                  << summedMilliseconds[stage] / TIMED_RUN_COUNT << std::endl;
    }
}

// Times the shared-memory denoise or the fused kernel (useFused picks which).
static void measureSingleKernel(bool useFused, DeviceBuffers& buffers, const ImageMetadata& metadata,
                                cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    dim3 blockShape(32, 8);
    dim3 gridShape = makeGridShape(metadata, 32, 8);
    float totalMilliseconds = 0.0f;

    for (int runNumber = 0; runNumber <= TIMED_RUN_COUNT; runNumber++) {
        checkCudaError(cudaEventRecord(startEvent), "recording start event");
        if (useFused) {
            fusedPipelineKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceOutput,
                                                           metadata.imageWidth, metadata.imageHeight,
                                                           metadata.blackLevel, metadata.whiteLevel,
                                                           metadata.whiteBalanceRed,
                                                           metadata.whiteBalanceGreen,
                                                           metadata.whiteBalanceBlue);
        } else {
            denoiseSharedKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                           metadata.imageWidth, metadata.imageHeight);
        }
        checkCudaError(cudaGetLastError(), "launching single kernel");
        checkCudaError(cudaEventRecord(stopEvent), "recording stop event");
        checkCudaError(cudaEventSynchronize(stopEvent), "waiting for single kernel");
        if (runNumber > 0) {
            totalMilliseconds = totalMilliseconds + elapsedMilliseconds(startEvent, stopEvent);
        }
    }
    if (useFused) {
        std::cout << "RESULT,fused_all," << totalMilliseconds / TIMED_RUN_COUNT << std::endl;
    } else {
        std::cout << "RESULT,shared_denoise," << totalMilliseconds / TIMED_RUN_COUNT << std::endl;
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
    std::cout << "RESULT,pixels," << pixelCount << std::endl;

    DeviceBuffers buffers;
    allocateDeviceBuffers(buffers, pixelCount);
    checkCudaError(cudaMemcpy(buffers.deviceBayer, bayerPixels.data(), buffers.bayerBytes,
                              cudaMemcpyHostToDevice), "copying Bayer data to the GPU");

    cudaEvent_t stageEvents[STAGE_COUNT + 1];
    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventCreate(&stageEvents[eventNumber]), "creating event");
    }

    measureCopyBandwidth(buffers, stageEvents[0], stageEvents[1]);
    measureStages("naive", false, 16, 16, buffers, metadata, stageEvents);
    measureStages("planar", true, 32, 8, buffers, metadata, stageEvents);
    // The planar run left a valid planar RGB image in deviceRgb for this one.
    measureSingleKernel(false, buffers, metadata, stageEvents[0], stageEvents[1]);
    measureSingleKernel(true, buffers, metadata, stageEvents[0], stageEvents[1]);

    for (int eventNumber = 0; eventNumber <= STAGE_COUNT; eventNumber++) {
        checkCudaError(cudaEventDestroy(stageEvents[eventNumber]), "destroying event");
    }
    freeDeviceBuffers(buffers);
    return 0;
}
