# view_output.py
# Usage: python view_output.py output/cpu_output.bin output/cpu_output.png
# Turns the raw 8-bit RGB file from C++/CUDA into a PNG you can look at.

import sys
import numpy
from PIL import Image

inputPath = sys.argv[1]
outputPath = sys.argv[2]

# Read width and height from the same metadata file the C++ program used.
metadata = {}
for line in open('data/metadata.txt'):
    parts = line.split()
    metadata[parts[0]] = float(parts[1])
imageWidth = int(metadata['width'])
imageHeight = int(metadata['height'])

rgbBytes = numpy.fromfile(inputPath, dtype=numpy.uint8)
expectedSize = imageWidth * imageHeight * 3
if rgbBytes.size != expectedSize:
    print('ERROR: file has', rgbBytes.size, 'bytes, expected', expectedSize)
    sys.exit(1)

rgbImage = rgbBytes.reshape(imageHeight, imageWidth, 3)
Image.fromarray(rgbImage).save(outputPath)

# Same 200x200 centre crop as the Bayer zoom in Milestone 0, for comparison.
centreRow = imageHeight // 2
centreColumn = imageWidth // 2
zoomCrop = rgbImage[centreRow:centreRow + 200, centreColumn:centreColumn + 200]
zoomPath = outputPath.replace('.png', '_zoom.png')
Image.fromarray(zoomCrop).resize((800, 800), Image.NEAREST).save(zoomPath)

print('Saved', outputPath, 'and', zoomPath)
