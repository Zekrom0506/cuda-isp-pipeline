// cpu_benchmark_main.cpp
// Milestone 8: the CPU pipeline from Milestone 1, run several times so the
// time is averaged like the GPU versions. Prints one machine-readable line
//   RESULT,cpu,<milliseconds>
// which the Python script collects. Also saves output/cpu_output.bin.

#include <chrono>
#include <iostream>
#include <vector>

#include "cpu_stages.h"
#include "isp_io.h"

// The CPU is slow, so fewer timed runs than on the GPU (3 instead of 10).
const int CPU_TIMED_RUN_COUNT = 3;

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

    // The vectors are created once, outside the loop. Run 0 is the warm-up:
    // it pays for the operating system setting up their memory (Milestone 1).
    std::vector<float> normalisedBayer;
    std::vector<float> rgbImage;
    std::vector<float> denoisedImage;
    std::vector<unsigned char> outputBytes;
    double totalMilliseconds = 0.0;

    for (int runNumber = 0; runNumber <= CPU_TIMED_RUN_COUNT; runNumber++) {
        std::chrono::steady_clock::time_point runStart = std::chrono::steady_clock::now();
        blackLevelCorrection(bayerPixels, metadata, normalisedBayer);
        demosaicBilinear(normalisedBayer, metadata.imageWidth, metadata.imageHeight, rgbImage);
        whiteBalance(rgbImage, metadata);
        denoiseGaussian3x3(rgbImage, metadata.imageWidth, metadata.imageHeight, denoisedImage);
        gammaAndConvertTo8Bit(denoisedImage, outputBytes);
        std::chrono::steady_clock::time_point runEnd = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::milli> runTime = runEnd - runStart;
        if (runNumber > 0) {
            totalMilliseconds = totalMilliseconds + runTime.count();
        }
    }

    double averageMilliseconds = totalMilliseconds / CPU_TIMED_RUN_COUNT;
    std::cout << "RESULT,cpu," << averageMilliseconds << std::endl;

    bool writeOk = writeRgbFile("output/cpu_output.bin", outputBytes);
    if (!writeOk) {
        return 1;
    }
    return 0;
}
