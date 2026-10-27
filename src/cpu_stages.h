// cpu_stages.h
// The five ISP stages, CPU version. One function per stage.
//
// Image layouts used between stages:
//   Bayer image: one float per pixel, index = row * imageWidth + column.
//   RGB image:   three floats per pixel (red, green, blue next to each other),
//                index = (row * imageWidth + column) * 3 + channel,
//                where channel is 0 for red, 1 for green, 2 for blue.

#pragma once

#include <cstdint>
#include <vector>

#include "isp_io.h"

// Stage 1 (cpu_pointwise_stages.cpp)
void blackLevelCorrection(const std::vector<uint16_t>& bayerPixels,
                          const ImageMetadata& metadata,
                          std::vector<float>& normalisedBayer);

// Stage 2 (cpu_neighbourhood_stages.cpp)
void demosaicBilinear(const std::vector<float>& normalisedBayer,
                      int imageWidth, int imageHeight,
                      std::vector<float>& rgbImage);

// Stage 3 (cpu_pointwise_stages.cpp) - changes rgbImage in place.
void whiteBalance(std::vector<float>& rgbImage, const ImageMetadata& metadata);

// Stage 4 (cpu_neighbourhood_stages.cpp)
void denoiseGaussian3x3(const std::vector<float>& rgbImage,
                        int imageWidth, int imageHeight,
                        std::vector<float>& denoisedImage);

// Stage 5 (cpu_pointwise_stages.cpp)
void gammaAndConvertTo8Bit(const std::vector<float>& rgbImage,
                           std::vector<unsigned char>& outputBytes);
