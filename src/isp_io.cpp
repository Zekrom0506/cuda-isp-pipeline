// isp_io.cpp
// File reading and writing. No image processing happens here.

#include "isp_io.h"

#include <fstream>
#include <iostream>

bool readMetadataFile(const std::string& metadataPath, ImageMetadata& metadata) {
    std::ifstream metadataFile(metadataPath);
    if (!metadataFile.is_open()) {
        std::cout << "ERROR: could not open " << metadataPath << std::endl;
        return false;
    }

    // The file has lines like "width 5796". We read a name, then a number,
    // until the file runs out. We count fields so we can detect a broken file.
    int fieldsFound = 0;
    std::string fieldName;
    float fieldValue = 0.0f;
    while (true) {
        metadataFile >> fieldName;
        metadataFile >> fieldValue;
        if (metadataFile.fail()) {
            break;
        }

        if (fieldName == "width") {
            metadata.imageWidth = static_cast<int>(fieldValue);
            fieldsFound = fieldsFound + 1;
        } else if (fieldName == "height") {
            metadata.imageHeight = static_cast<int>(fieldValue);
            fieldsFound = fieldsFound + 1;
        } else if (fieldName == "black_level") {
            metadata.blackLevel = fieldValue;
            fieldsFound = fieldsFound + 1;
        } else if (fieldName == "white_level") {
            metadata.whiteLevel = fieldValue;
            fieldsFound = fieldsFound + 1;
        } else if (fieldName == "wb_red") {
            metadata.whiteBalanceRed = fieldValue;
            fieldsFound = fieldsFound + 1;
        } else if (fieldName == "wb_green") {
            metadata.whiteBalanceGreen = fieldValue;
            fieldsFound = fieldsFound + 1;
        } else if (fieldName == "wb_blue") {
            metadata.whiteBalanceBlue = fieldValue;
            fieldsFound = fieldsFound + 1;
        }
    }

    if (fieldsFound != 7) {
        std::cout << "ERROR: metadata file has " << fieldsFound
                  << " of the 7 expected fields" << std::endl;
        return false;
    }
    return true;
}

bool readBayerFile(const std::string& bayerPath, int pixelCount,
                   std::vector<uint16_t>& bayerPixels) {
    // std::ios::binary means "give me the raw bytes, do not treat this as text".
    std::ifstream bayerFile(bayerPath, std::ios::binary);
    if (!bayerFile.is_open()) {
        std::cout << "ERROR: could not open " << bayerPath << std::endl;
        return false;
    }

    // Make room for every pixel before reading into the vector.
    bayerPixels.resize(pixelCount);

    // Python saved little-endian 16-bit numbers. The CPUs we use (Colab's x86)
    // are also little-endian, so the bytes can be copied straight into memory.
    // read() wants a char pointer, so we reinterpret our uint16_t array as bytes.
    long long expectedByteCount = static_cast<long long>(pixelCount) * 2;
    char* destinationBytes = reinterpret_cast<char*>(bayerPixels.data());
    bayerFile.read(destinationBytes, expectedByteCount);

    long long actualByteCount = bayerFile.gcount();
    if (actualByteCount != expectedByteCount) {
        std::cout << "ERROR: expected " << expectedByteCount << " bytes but read "
                  << actualByteCount << std::endl;
        return false;
    }
    return true;
}

bool writeRgbFile(const std::string& outputPath,
                  const std::vector<unsigned char>& rgbBytes) {
    std::ofstream outputFile(outputPath, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cout << "ERROR: could not create " << outputPath << std::endl;
        return false;
    }

    // The output is just R,G,B bytes for each pixel, row by row. Python reshapes it.
    const char* sourceBytes = reinterpret_cast<const char*>(rgbBytes.data());
    long long byteCount = static_cast<long long>(rgbBytes.size());
    outputFile.write(sourceBytes, byteCount);
    return true;
}
