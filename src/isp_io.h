// isp_io.h
// Reading and writing the simple files our pipeline uses.
// Shared by the CPU program now and the CUDA programs later.

// "#pragma once" tells the compiler: if this file is included twice,
// only use it the first time. It prevents "defined twice" errors.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// All the numbers from metadata.txt, kept together in one place
// so we can pass them to functions as a single argument.
struct ImageMetadata {
    int imageWidth;
    int imageHeight;
    float blackLevel;
    float whiteLevel;
    float whiteBalanceRed;
    float whiteBalanceGreen;
    float whiteBalanceBlue;
};

// Each function returns true on success and false on failure,
// so main() can stop with a clear message instead of crashing.
bool readMetadataFile(const std::string& metadataPath, ImageMetadata& metadata);

bool readBayerFile(const std::string& bayerPath, int pixelCount,
                   std::vector<uint16_t>& bayerPixels);

bool writeRgbFile(const std::string& outputPath,
                  const std::vector<unsigned char>& rgbBytes);
