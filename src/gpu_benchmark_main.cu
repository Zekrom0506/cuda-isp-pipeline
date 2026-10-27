// gpu_benchmark_main.cu
// Milestone 8: times every GPU version on the image in ./data and prints
// machine-readable lines for the Python script:
//   RESULT,<version>,<milliseconds>
// Versions (all names from earlier milestones):
//   naive  - Milestone 2/4: interleaved RGB, 16x16 blocks, 5 kernels
//   planar - Milestone 5 config E: planar RGB, 32x8 blocks, 5 kernels
//   shared - Milestone 6: like planar, but shared-memory denoise
//   fused  - Milestone 7: one kernel for all 5 stages
//   copy_in / copy_out - CPU<->GPU copy times (the same for every version)
// Each version: 1 warm-up run + 10 averaged runs (Milestone 4 method).

#include <iostream>
#include <string>
#include <vector>

#include "cuda_helpers.h"
#include "gpu_buffers.h"
#include "gpu_fused_kernel.h"
#include "gpu_kernels.h"
#include "gpu_planar_kernels.h"
#include "gpu_shared_denoise.h"
#include "isp_io.h"

const int TIMED_RUN_COUNT = 10;
const int VERSION_NAIVE = 0;
const int VERSION_PLANAR = 1;
const int VERSION_SHARED = 2;
const int VERSION_FUSED = 3;
const int VERSION_COUNT = 4;

// Makes enough blocks of the given shape to cover the whole image.
static dim3 makeGridShape(const ImageMetadata& metadata, dim3 blockShape) {
    // dim3 stores unsigned numbers; convert to int so all the maths uses one type.
    int blockWidth = static_cast<int>(blockShape.x);
    int blockHeight = static_cast<int>(blockShape.y);
    int blocksAcross = (metadata.imageWidth + blockWidth - 1) / blockWidth;
    int blocksDown = (metadata.imageHeight + blockHeight - 1) / blockHeight;
    dim3 gridShape(blocksAcross, blocksDown);
    return gridShape;
}

static void launchNaive(DeviceBuffers& buffers, const ImageMetadata& metadata) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;
    dim3 blockShape(16, 16);
    dim3 gridShape = makeGridShape(metadata, blockShape);

    blackLevelKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceNormalised,
                                                imageWidth, imageHeight,
                                                metadata.blackLevel, metadata.whiteLevel);
    checkCudaError(cudaGetLastError(), "launching blackLevelKernel");
    demosaicKernel<<<gridShape, blockShape>>>(buffers.deviceNormalised, buffers.deviceRgb,
                                              imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching demosaicKernel");
    whiteBalanceKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, imageWidth, imageHeight,
                                                  metadata.whiteBalanceRed,
                                                  metadata.whiteBalanceGreen,
                                                  metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching whiteBalanceKernel");
    denoiseKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                             imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching denoiseKernel");
    gammaKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                           imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching gammaKernel");
}

// Planar 5-kernel pipeline. useSharedDenoise picks the Milestone 6 denoise.
static void launchPlanar(DeviceBuffers& buffers, const ImageMetadata& metadata, bool useSharedDenoise) {
    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;
    dim3 blockShape(32, 8);
    dim3 gridShape = makeGridShape(metadata, blockShape);

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
    if (useSharedDenoise) {
        denoiseSharedKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                       imageWidth, imageHeight);
    } else {
        denoisePlanarKernel<<<gridShape, blockShape>>>(buffers.deviceRgb, buffers.deviceDenoised,
                                                       imageWidth, imageHeight);
    }
    checkCudaError(cudaGetLastError(), "launching denoise");
    gammaPlanarKernel<<<gridShape, blockShape>>>(buffers.deviceDenoised, buffers.deviceOutput,
                                                 imageWidth, imageHeight);
    checkCudaError(cudaGetLastError(), "launching gammaPlanarKernel");
}

static void launchFused(DeviceBuffers& buffers, const ImageMetadata& metadata) {
    dim3 blockShape(FUSED_TILE_WIDTH, FUSED_TILE_HEIGHT);
    dim3 gridShape = makeGridShape(metadata, blockShape);
    fusedPipelineKernel<<<gridShape, blockShape>>>(buffers.deviceBayer, buffers.deviceOutput,
                                                   metadata.imageWidth, metadata.imageHeight,
                                                   metadata.blackLevel, metadata.whiteLevel,
                                                   metadata.whiteBalanceRed,
                                                   metadata.whiteBalanceGreen,
                                                   metadata.whiteBalanceBlue);
    checkCudaError(cudaGetLastError(), "launching fusedPipelineKernel");
}

static void launchVersion(int version, DeviceBuffers& buffers, const ImageMetadata& metadata) {
    if (version == VERSION_NAIVE) {
        launchNaive(buffers, metadata);
    } else if (version == VERSION_PLANAR) {
        launchPlanar(buffers, metadata, false);
    } else if (version == VERSION_SHARED) {
        launchPlanar(buffers, metadata, true);
    } else {
        launchFused(buffers, metadata);
    }
}

static float averageMilliseconds(int version, DeviceBuffers& buffers, const ImageMetadata& metadata,
                                 cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    float totalMilliseconds = 0.0f;
    for (int runNumber = 0; runNumber <= TIMED_RUN_COUNT; runNumber++) {
        checkCudaError(cudaEventRecord(startEvent), "recording start event");
        launchVersion(version, buffers, metadata);
        checkCudaError(cudaEventRecord(stopEvent), "recording stop event");
        checkCudaError(cudaEventSynchronize(stopEvent), "waiting for the run");
        float runMilliseconds = 0.0f;
        checkCudaError(cudaEventElapsedTime(&runMilliseconds, startEvent, stopEvent),
                       "reading run time");
        if (runNumber > 0) {
            totalMilliseconds = totalMilliseconds + runMilliseconds;
        }
    }
    return totalMilliseconds / TIMED_RUN_COUNT;
}

// Times one cudaMemcpy, averaged the same way as the kernels.
static float averageCopyMilliseconds(void* destination, const void* source, size_t byteCount,
                                     cudaMemcpyKind direction,
                                     cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    float totalMilliseconds = 0.0f;
    for (int runNumber = 0; runNumber <= TIMED_RUN_COUNT; runNumber++) {
        checkCudaError(cudaEventRecord(startEvent), "recording start event");
        checkCudaError(cudaMemcpy(destination, source, byteCount, direction), "copying");
        checkCudaError(cudaEventRecord(stopEvent), "recording stop event");
        checkCudaError(cudaEventSynchronize(stopEvent), "waiting for the copy");
        float runMilliseconds = 0.0f;
        checkCudaError(cudaEventElapsedTime(&runMilliseconds, startEvent, stopEvent),
                       "reading copy time");
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
    std::vector<unsigned char> outputBytes(buffers.outputBytes);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    checkCudaError(cudaEventCreate(&startEvent), "creating start event");
    checkCudaError(cudaEventCreate(&stopEvent), "creating stop event");

    // Copy in first (also leaves the Bayer data on the GPU for the kernels).
    float copyInMilliseconds = averageCopyMilliseconds(buffers.deviceBayer, bayerPixels.data(),
                                                       buffers.bayerBytes, cudaMemcpyHostToDevice,
                                                       startEvent, stopEvent);
    std::cout << "RESULT,copy_in," << copyInMilliseconds << std::endl;

    const char* versionNames[VERSION_COUNT] = {"naive", "planar", "shared", "fused"};
    for (int version = 0; version < VERSION_COUNT; version++) {
        float kernelMilliseconds = averageMilliseconds(version, buffers, metadata, startEvent, stopEvent);
        std::cout << "RESULT," << versionNames[version] << "," << kernelMilliseconds << std::endl;
    }

    // The last version run was "fused": time copying its output back, and save it.
    float copyOutMilliseconds = averageCopyMilliseconds(outputBytes.data(), buffers.deviceOutput,
                                                        buffers.outputBytes, cudaMemcpyDeviceToHost,
                                                        startEvent, stopEvent);
    std::cout << "RESULT,copy_out," << copyOutMilliseconds << std::endl;

    checkCudaError(cudaEventDestroy(startEvent), "destroying start event");
    checkCudaError(cudaEventDestroy(stopEvent), "destroying stop event");
    freeDeviceBuffers(buffers);

    bool writeOk = writeRgbFile("output/gpu_fused_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    return 0;
}
