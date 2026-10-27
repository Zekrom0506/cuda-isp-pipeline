// compare_outputs.cpp
// Compares two 8-bit RGB output files value by value.
// Usage: compare_outputs referenceFile testFile
// Example: compare_outputs output/cpu_output.bin output/gpu_naive_output.bin
// We will reuse this for every GPU version in later milestones.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "isp_io.h"

// Reads a whole binary file into a byte vector. Returns false on failure.
static bool readWholeFile(const char* filePath, std::vector<unsigned char>& fileBytes) {
    // std::ios::ate opens the file with the read position at the END,
    // so tellg() tells us the file size straight away.
    std::ifstream inputFile(filePath, std::ios::binary | std::ios::ate);
    if (!inputFile.is_open()) {
        std::cout << "ERROR: could not open " << filePath << std::endl;
        return false;
    }
    long long fileSize = static_cast<long long>(inputFile.tellg());
    inputFile.seekg(0);

    fileBytes.resize(fileSize);
    char* destinationBytes = reinterpret_cast<char*>(fileBytes.data());
    inputFile.read(destinationBytes, fileSize);
    return true;
}

// argc is how many words were typed on the command line (including the
// program name), and argv holds those words. argv[1] and argv[2] are our files.
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: compare_outputs referenceFile testFile" << std::endl;
        return 1;
    }

    ImageMetadata metadata;
    bool metadataOk = readMetadataFile("data/metadata.txt", metadata);
    if (!metadataOk) {
        return 1;
    }
    int imageWidth = metadata.imageWidth;

    std::vector<unsigned char> referenceBytes;
    std::vector<unsigned char> testBytes;
    bool referenceOk = readWholeFile(argv[1], referenceBytes);
    bool testOk = readWholeFile(argv[2], testBytes);
    if (!referenceOk || !testOk) {
        return 1;
    }
    if (referenceBytes.size() != testBytes.size()) {
        std::cout << "ERROR: files have different sizes" << std::endl;
        return 1;
    }

    // Walk through every value. Record the largest difference, where it was,
    // and how many values differ by each amount (0, 1, 2, ... up to 255).
    long long valueCount = static_cast<long long>(referenceBytes.size());
    std::vector<long long> countPerDifference(256, 0);
    int largestDifference = 0;
    long long largestDifferenceIndex = 0;

    for (long long valueIndex = 0; valueIndex < valueCount; valueIndex++) {
        int referenceValue = referenceBytes[valueIndex];
        int testValue = testBytes[valueIndex];
        int difference = std::abs(referenceValue - testValue);

        countPerDifference[difference] = countPerDifference[difference] + 1;

        if (difference > largestDifference) {
            largestDifference = difference;
            largestDifferenceIndex = valueIndex;
        }
    }

    // Turn the index back into (row, column, channel) so a problem can be located.
    long long pixelIndex = largestDifferenceIndex / 3;
    int channel = static_cast<int>(largestDifferenceIndex % 3);
    long long row = pixelIndex / imageWidth;
    long long column = pixelIndex % imageWidth;

    std::cout << "Values compared      : " << valueCount << std::endl;
    std::cout << "Largest difference   : " << largestDifference << " (out of 255)" << std::endl;
    if (largestDifference > 0) {
        std::cout << "  first seen at row " << row << ", column " << column
                  << ", channel " << channel << " (0=red 1=green 2=blue)" << std::endl;
    }

    std::cout << "How many values differ by each amount:" << std::endl;
    for (int difference = 0; difference <= largestDifference; difference++) {
        double percent = 100.0 * countPerDifference[difference] / valueCount;
        std::cout << "  differ by " << difference << ": " << countPerDifference[difference]
                  << " values (" << percent << " %)" << std::endl;
    }

    // A difference of 1 can come from harmless floating-point rounding (see the
    // notebook). Anything bigger means the two programs do different maths.
    if (largestDifference <= 1) {
        std::cout << "RESULT: PASS" << std::endl;
        return 0;
    } else {
        std::cout << "RESULT: FAIL - look for a bug near the location above" << std::endl;
        return 1;
    }
}
