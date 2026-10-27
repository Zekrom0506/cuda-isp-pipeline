// cpu_main.cpp
// CPU reference pipeline. Reads data/bayer.bin, runs the 5 stages, times each
// one, and writes output/cpu_output.bin (8-bit RGB). Every GPU version later
// will be checked against this output.

#include <chrono>
#include <iostream>
#include <vector>

#include "cpu_stages.h"
#include "isp_io.h"

// A shorter name for the clock's time type, so the timing lines stay readable.
// steady_clock is used because it never jumps (unlike the wall clock).
using TimePoint = std::chrono::steady_clock::time_point;

static TimePoint currentTime() {
    return std::chrono::steady_clock::now();
}

static double millisecondsBetween(TimePoint startTime, TimePoint endTime) {
    std::chrono::duration<double, std::milli> elapsed = endTime - startTime;
    return elapsed.count();
}

int main() {
    // ---- Load input (not timed: file reading is not part of the pipeline) ----
    ImageMetadata metadata;
    bool metadataOk = readMetadataFile("data/metadata.txt", metadata);
    if (!metadataOk) {
        return 1;
    }

    int imageWidth = metadata.imageWidth;
    int imageHeight = metadata.imageHeight;
    int pixelCount = imageWidth * imageHeight;
    std::cout << "Image: " << imageWidth << " x " << imageHeight
              << " (" << pixelCount << " pixels)" << std::endl;

    std::vector<uint16_t> bayerPixels;
    bool bayerOk = readBayerFile("data/bayer.bin", pixelCount, bayerPixels);
    if (!bayerOk) {
        return 1;
    }

    // ---- Run and time each stage ----
    std::vector<float> normalisedBayer;
    std::vector<float> rgbImage;
    std::vector<float> denoisedImage;
    std::vector<unsigned char> outputBytes;

    TimePoint pipelineStart = currentTime();

    TimePoint stageStart = currentTime();
    blackLevelCorrection(bayerPixels, metadata, normalisedBayer);
    double blackLevelMs = millisecondsBetween(stageStart, currentTime());

    stageStart = currentTime();
    demosaicBilinear(normalisedBayer, imageWidth, imageHeight, rgbImage);
    double demosaicMs = millisecondsBetween(stageStart, currentTime());

    stageStart = currentTime();
    whiteBalance(rgbImage, metadata);
    double whiteBalanceMs = millisecondsBetween(stageStart, currentTime());

    stageStart = currentTime();
    denoiseGaussian3x3(rgbImage, imageWidth, imageHeight, denoisedImage);
    double denoiseMs = millisecondsBetween(stageStart, currentTime());

    stageStart = currentTime();
    gammaAndConvertTo8Bit(denoisedImage, outputBytes);
    double gammaMs = millisecondsBetween(stageStart, currentTime());

    double totalMs = millisecondsBetween(pipelineStart, currentTime());

    std::cout << "Stage 1 black level : " << blackLevelMs << " ms" << std::endl;
    std::cout << "Stage 2 demosaic    : " << demosaicMs << " ms" << std::endl;
    std::cout << "Stage 3 white bal.  : " << whiteBalanceMs << " ms" << std::endl;
    std::cout << "Stage 4 denoise     : " << denoiseMs << " ms" << std::endl;
    std::cout << "Stage 5 gamma       : " << gammaMs << " ms" << std::endl;
    std::cout << "Total pipeline      : " << totalMs << " ms" << std::endl;

    // ---- Save output (not timed) ----
    bool writeOk = writeRgbFile("output/cpu_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    std::cout << "Saved output/cpu_output.bin" << std::endl;
    return 0;
}
