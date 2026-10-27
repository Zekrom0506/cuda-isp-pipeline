// cpu_neighbourhood_stages.cpp
// Stages 2 and 4. These are "neighbourhood" stages: each output pixel needs
// the 3x3 block of input pixels around it. They read 9 values per pixel, which
// is why they will be the most interesting stages to optimise on the GPU.

#include "cpu_stages.h"

// Border handling for the Bayer image: "reflect".
// A pixel on the edge has missing neighbours outside the image. For the Bayer
// image we must replace them with a pixel of the SAME COLOUR. Reflecting by one
// (index -1 becomes 1, index size becomes size-2) moves by 2 pixels, and since
// the Bayer pattern repeats every 2 pixels, the colour stays the same.
// (Clamping -1 to 0 would give a pixel of a DIFFERENT colour - wrong.)
static int reflectIndex(int index, int size) {
    if (index < 0) {
        return -index;
    }
    if (index >= size) {
        return 2 * size - 2 - index;
    }
    return index;
}

// Border handling for the RGB image: "clamp" (use the nearest edge pixel).
// After demosaic every pixel has all three colours, so any neighbour works.
static int clampIndex(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static float readBayerValue(const std::vector<float>& bayerImage,
                            int imageWidth, int imageHeight, int row, int column) {
    int safeRow = reflectIndex(row, imageHeight);
    int safeColumn = reflectIndex(column, imageWidth);
    int pixelIndex = safeRow * imageWidth + safeColumn;
    return bayerImage[pixelIndex];
}

// Stage 2: bilinear demosaic for an RGGB sensor.
// The 2x2 tile is      R G      (even row, even column) = red
//                      G B      (odd row, odd column)   = blue
// Every pixel knows one colour. The two missing colours are the average of the
// nearest neighbours that DO know that colour:
//   at red:    green = average of up/down/left/right, blue = average of 4 diagonals
//   at blue:   green = average of up/down/left/right, red  = average of 4 diagonals
//   at green on a red row:  red = average of left/right, blue = average of up/down
//   at green on a blue row: red = average of up/down,  blue = average of left/right
void demosaicBilinear(const std::vector<float>& normalisedBayer,
                      int imageWidth, int imageHeight,
                      std::vector<float>& rgbImage) {
    rgbImage.resize(static_cast<size_t>(imageWidth) * imageHeight * 3);

    for (int row = 0; row < imageHeight; row++) {
        for (int column = 0; column < imageWidth; column++) {
            // Read the full 3x3 neighbourhood once. Not every value is used for
            // every pixel colour, but reading all 9 keeps the code uniform.
            float centre = readBayerValue(normalisedBayer, imageWidth, imageHeight, row, column);
            float up = readBayerValue(normalisedBayer, imageWidth, imageHeight, row - 1, column);
            float down = readBayerValue(normalisedBayer, imageWidth, imageHeight, row + 1, column);
            float left = readBayerValue(normalisedBayer, imageWidth, imageHeight, row, column - 1);
            float right = readBayerValue(normalisedBayer, imageWidth, imageHeight, row, column + 1);
            float upLeft = readBayerValue(normalisedBayer, imageWidth, imageHeight, row - 1, column - 1);
            float upRight = readBayerValue(normalisedBayer, imageWidth, imageHeight, row - 1, column + 1);
            float downLeft = readBayerValue(normalisedBayer, imageWidth, imageHeight, row + 1, column - 1);
            float downRight = readBayerValue(normalisedBayer, imageWidth, imageHeight, row + 1, column + 1);

            float crossAverage = (up + down + left + right) / 4.0f;
            float diagonalAverage = (upLeft + upRight + downLeft + downRight) / 4.0f;
            float horizontalAverage = (left + right) / 2.0f;
            float verticalAverage = (up + down) / 2.0f;

            bool isEvenRow = (row % 2 == 0);
            bool isEvenColumn = (column % 2 == 0);

            float redValue = 0.0f;
            float greenValue = 0.0f;
            float blueValue = 0.0f;

            if (isEvenRow && isEvenColumn) {
                // Red pixel.
                redValue = centre;
                greenValue = crossAverage;
                blueValue = diagonalAverage;
            } else if (isEvenRow && !isEvenColumn) {
                // Green pixel on a red row: red is left/right, blue is up/down.
                redValue = horizontalAverage;
                greenValue = centre;
                blueValue = verticalAverage;
            } else if (!isEvenRow && isEvenColumn) {
                // Green pixel on a blue row: red is up/down, blue is left/right.
                redValue = verticalAverage;
                greenValue = centre;
                blueValue = horizontalAverage;
            } else {
                // Blue pixel.
                redValue = diagonalAverage;
                greenValue = crossAverage;
                blueValue = centre;
            }

            size_t pixelIndex = static_cast<size_t>(row) * imageWidth + column;
            rgbImage[pixelIndex * 3 + 0] = redValue;
            rgbImage[pixelIndex * 3 + 1] = greenValue;
            rgbImage[pixelIndex * 3 + 2] = blueValue;
        }
    }
}

// Stage 4: 3x3 Gaussian blur on each colour channel.
// Weights:   1 2 1
//            2 4 2     divided by 16 (the sum), so brightness is unchanged.
//            1 2 1
// The centre counts most, close neighbours less. Random noise differs from
// pixel to pixel, so averaging cancels some of it while edges stay fairly sharp.
// The output MUST be a separate buffer: if we wrote into the input, later
// pixels would average already-blurred values.
void denoiseGaussian3x3(const std::vector<float>& rgbImage,
                        int imageWidth, int imageHeight,
                        std::vector<float>& denoisedImage) {
    denoisedImage.resize(rgbImage.size());

    const float gaussianWeights[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };
    const float weightSum = 16.0f;

    for (int row = 0; row < imageHeight; row++) {
        for (int column = 0; column < imageWidth; column++) {
            for (int channel = 0; channel < 3; channel++) {
                float weightedTotal = 0.0f;

                for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
                    for (int columnOffset = -1; columnOffset <= 1; columnOffset++) {
                        int neighbourRow = clampIndex(row + rowOffset, imageHeight);
                        int neighbourColumn = clampIndex(column + columnOffset, imageWidth);
                        size_t neighbourPixel = static_cast<size_t>(neighbourRow) * imageWidth + neighbourColumn;
                        float neighbourValue = rgbImage[neighbourPixel * 3 + channel];
                        float weight = gaussianWeights[rowOffset + 1][columnOffset + 1];
                        weightedTotal = weightedTotal + neighbourValue * weight;
                    }
                }

                size_t pixelIndex = static_cast<size_t>(row) * imageWidth + column;
                denoisedImage[pixelIndex * 3 + channel] = weightedTotal / weightSum;
            }
        }
    }
}
