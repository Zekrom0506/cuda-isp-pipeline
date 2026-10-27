# convert_raw.py
# Usage: python convert_raw.py data/yourfile.dng
# Output: data/bayer.bin and data/metadata.txt

import sys
import numpy
import rawpy


def describeBayerPattern(rawFile):
    # raw_pattern is a 2x2 array of colour indices for the top-left tile.
    # color_desc tells us what each index means, e.g. b'RGBG'.
    # The fourth entry is a second green, so we map it to plain 'G'.
    colourNames = rawFile.color_desc.decode('ascii')
    patternText = ''
    for rowInTile in range(2):
        for columnInTile in range(2):
            colourIndex = rawFile.raw_pattern[rowInTile][columnInTile]
            patternText = patternText + colourNames[colourIndex]
    return patternText


def cropToRGGB(bayerImage, patternText):
    # Decide how many rows/columns to drop from the top-left (see explanation above).
    rowsToDropAtTop = 0
    columnsToDropAtLeft = 0
    if patternText == 'RGGB':
        rowsToDropAtTop = 0
        columnsToDropAtLeft = 0
    elif patternText == 'GRBG':
        columnsToDropAtLeft = 1
    elif patternText == 'GBRG':
        rowsToDropAtTop = 1
    elif patternText == 'BGGR':
        rowsToDropAtTop = 1
        columnsToDropAtLeft = 1
    else:
        print('Unsupported pattern (not a standard Bayer sensor):', patternText)
        sys.exit(1)

    croppedImage = bayerImage[rowsToDropAtTop:, columnsToDropAtLeft:]

    # Make width and height even so every 2x2 tile is complete.
    imageHeight = croppedImage.shape[0]
    imageWidth = croppedImage.shape[1]
    evenHeight = imageHeight - (imageHeight % 2)
    evenWidth = imageWidth - (imageWidth % 2)
    croppedImage = croppedImage[:evenHeight, :evenWidth]

    print('Cropped', rowsToDropAtTop, 'row(s) at top and', columnsToDropAtLeft, 'column(s) at left')
    return croppedImage


def main():
    inputPath = sys.argv[1]
    rawFile = rawpy.imread(inputPath)

    # raw_image_visible excludes the hidden 'masked' border pixels some sensors have.
    bayerImage = rawFile.raw_image_visible.copy()

    patternText = describeBayerPattern(rawFile)
    print('Original Bayer pattern:', patternText)
    print('Original size (width x height):', bayerImage.shape[1], 'x', bayerImage.shape[0])

    bayerImage = cropToRGGB(bayerImage, patternText)
    imageHeight = bayerImage.shape[0]
    imageWidth = bayerImage.shape[1]

    # Black level: the value a pixel reports in total darkness.
    # Some cameras give one per channel; they are almost always equal, so we use the average.
    blackLevelPerChannel = rawFile.black_level_per_channel
    blackLevel = float(numpy.mean(blackLevelPerChannel))

    # White level: the value of a fully saturated pixel.
    whiteLevel = float(rawFile.white_level)

    # White balance multipliers chosen by the camera when the photo was taken: [R, G, B, G2].
    # We divide by green so green = 1.0, which is the usual convention.
    cameraWhiteBalance = rawFile.camera_whitebalance
    greenMultiplier = cameraWhiteBalance[1]
    redGain = cameraWhiteBalance[0] / greenMultiplier
    greenGain = 1.0
    blueGain = cameraWhiteBalance[2] / greenMultiplier

    # Save pixels as little-endian unsigned 16-bit, row by row.
    bayerAsUint16 = numpy.ascontiguousarray(bayerImage, dtype='<u2')
    bayerAsUint16.tofile('data/bayer.bin')

    # Simple 'name value' lines, easy to read from C++ with ifstream >>.
    metadataFile = open('data/metadata.txt', 'w')
    metadataFile.write('width ' + str(imageWidth) + '\n')
    metadataFile.write('height ' + str(imageHeight) + '\n')
    metadataFile.write('black_level ' + str(blackLevel) + '\n')
    metadataFile.write('white_level ' + str(whiteLevel) + '\n')
    metadataFile.write('wb_red ' + str(redGain) + '\n')
    metadataFile.write('wb_green ' + str(greenGain) + '\n')
    metadataFile.write('wb_blue ' + str(blueGain) + '\n')
    metadataFile.close()

    print('Black level per channel:', blackLevelPerChannel)
    print('Pixel value range in file:', int(bayerImage.min()), 'to', int(bayerImage.max()))
    print('Saved data/bayer.bin and data/metadata.txt')
    print(open('data/metadata.txt').read())


main()
