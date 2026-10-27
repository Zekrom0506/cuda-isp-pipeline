// cpu_pointwise_stages.cpp
// Stages 1, 3 and 5. These are "pointwise": each output value depends only on
// the SAME pixel of the input, never on its neighbours. That makes them the
// easiest stages to move to the GPU later (one thread per pixel, no sharing).

#include "cpu_stages.h"

#include <cmath>

// Stage 1: black level correction.
// The sensor never reports 0 in darkness; it reports about blackLevel.
// Subtracting it makes "no light" equal 0. Dividing by (white - black) makes
// "fully saturated" equal 1. After this, every value is in 0.0 to 1.0.
void blackLevelCorrection(const std::vector<uint16_t>& bayerPixels,
                          const ImageMetadata& metadata,
                          std::vector<float>& normalisedBayer) {
    int pixelCount = metadata.imageWidth * metadata.imageHeight;
    normalisedBayer.resize(pixelCount);

    float usefulRange = metadata.whiteLevel - metadata.blackLevel;

    for (int pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++) {
        float rawValue = static_cast<float>(bayerPixels[pixelIndex]);
        float normalisedValue = (rawValue - metadata.blackLevel) / usefulRange;

        // Noise can push dark pixels slightly below black (your file has 2023 < 2047),
        // and blown highlights can be above white (15464 > 15000). Clamp both.
        if (normalisedValue < 0.0f) {
            normalisedValue = 0.0f;
        }
        if (normalisedValue > 1.0f) {
            normalisedValue = 1.0f;
        }

        normalisedBayer[pixelIndex] = normalisedValue;
    }
}

// Stage 3: white balance.
// The sensor is more sensitive to green than to red or blue, so a white wall
// comes out greenish. Multiplying red and blue by the camera's gains makes
// white objects have equal red, green and blue again.
void whiteBalance(std::vector<float>& rgbImage, const ImageMetadata& metadata) {
    int pixelCount = static_cast<int>(rgbImage.size() / 3);

    for (int pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++) {
        int redIndex = pixelIndex * 3 + 0;
        int greenIndex = pixelIndex * 3 + 1;
        int blueIndex = pixelIndex * 3 + 2;

        float redValue = rgbImage[redIndex] * metadata.whiteBalanceRed;
        float greenValue = rgbImage[greenIndex] * metadata.whiteBalanceGreen;
        float blueValue = rgbImage[blueIndex] * metadata.whiteBalanceBlue;

        // A gain above 1 can push bright values past 1.0. Clamp so saturated
        // areas become white instead of an out-of-range colour.
        if (redValue > 1.0f) {
            redValue = 1.0f;
        }
        if (greenValue > 1.0f) {
            greenValue = 1.0f;
        }
        if (blueValue > 1.0f) {
            blueValue = 1.0f;
        }

        rgbImage[redIndex] = redValue;
        rgbImage[greenIndex] = greenValue;
        rgbImage[blueIndex] = blueValue;
    }
}

// Stage 5: gamma correction and conversion to 8-bit.
// Sensor values are "linear" (twice the light = twice the number), but screens
// and our eyes expect a curve. Raising to the power 1/2.2 brightens the dark
// tones so the photo looks natural. Then we scale 0..1 up to 0..255.
void gammaAndConvertTo8Bit(const std::vector<float>& rgbImage,
                           std::vector<unsigned char>& outputBytes) {
    int valueCount = static_cast<int>(rgbImage.size());
    outputBytes.resize(valueCount);

    float gammaExponent = 1.0f / 2.2f;

    for (int valueIndex = 0; valueIndex < valueCount; valueIndex++) {
        float linearValue = rgbImage[valueIndex];
        if (linearValue < 0.0f) {
            linearValue = 0.0f;
        }
        if (linearValue > 1.0f) {
            linearValue = 1.0f;
        }

        float gammaValue = std::pow(linearValue, gammaExponent);

        // Adding 0.5 before converting rounds to the nearest whole number
        // instead of always rounding down.
        float scaledValue = gammaValue * 255.0f + 0.5f;
        outputBytes[valueIndex] = static_cast<unsigned char>(scaledValue);
    }
}
