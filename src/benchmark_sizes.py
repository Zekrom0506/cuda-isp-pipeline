# benchmark_sizes.py
# Milestone 8: run every pipeline version on several image sizes.
#
# For each size it:
#   1. crops the full Bayer image (from data/) and saves it in its own folder,
#      benchmark/size_<width>x<height>/data/  (bayer.bin + metadata.txt)
#   2. runs the CPU benchmark and the GPU benchmark INSIDE that folder, so the
#      C++ programs read "data/..." and write "output/..." exactly as before
#   3. checks the fused GPU output against the CPU output with compare_outputs
# Then it saves benchmark/results.csv, prints a table, and draws a chart.

import os
import subprocess
import numpy
import matplotlib.pyplot as pyplot

# Sizes to test. Every width and height is EVEN, and every crop starts at an
# even row and column, so the top-left pixel is still red (RGGB) - the same
# reason as the one-pixel crop in Milestone 0.
cropSizes = [
    (640, 428),
    (1280, 856),
    (2560, 1710),
    (3840, 2564),
    (5796, 3870),
]

versionNames = ['cpu', 'naive', 'planar', 'shared', 'fused']


def readMetadata(metadataPath):
    metadata = {}
    for line in open(metadataPath):
        parts = line.split()
        metadata[parts[0]] = parts[1]
    return metadata


def makeEven(number):
    return number - (number % 2)


def writeCrop(fullBayer, fullMetadata, cropWidth, cropHeight):
    # Take the crop from the centre, so small crops still show the subject.
    fullHeight = fullBayer.shape[0]
    fullWidth = fullBayer.shape[1]
    startRow = makeEven((fullHeight - cropHeight) // 2)
    startColumn = makeEven((fullWidth - cropWidth) // 2)
    croppedBayer = fullBayer[startRow:startRow + cropHeight, startColumn:startColumn + cropWidth]

    sizeFolder = 'benchmark/size_' + str(cropWidth) + 'x' + str(cropHeight)
    os.makedirs(sizeFolder + '/data', exist_ok=True)
    os.makedirs(sizeFolder + '/output', exist_ok=True)

    numpy.ascontiguousarray(croppedBayer, dtype='<u2').tofile(sizeFolder + '/data/bayer.bin')

    # Same metadata as the full image, only width and height change.
    metadataFile = open(sizeFolder + '/data/metadata.txt', 'w')
    for fieldName in ['width', 'height', 'black_level', 'white_level', 'wb_red', 'wb_green', 'wb_blue']:
        fieldValue = fullMetadata[fieldName]
        if fieldName == 'width':
            fieldValue = str(cropWidth)
        if fieldName == 'height':
            fieldValue = str(cropHeight)
        metadataFile.write(fieldName + ' ' + fieldValue + '\n')
    metadataFile.close()
    return sizeFolder


def runProgram(programPath, workingFolder):
    # Runs a compiled program inside workingFolder and returns its printed text.
    completedRun = subprocess.run([programPath], cwd=workingFolder, capture_output=True, text=True)
    if completedRun.returncode != 0:
        print('ERROR running', programPath, 'in', workingFolder)
        print(completedRun.stdout)
        print(completedRun.stderr)
    return completedRun.stdout


def parseResults(programOutput, resultsForThisSize):
    # Lines look like "RESULT,fused,3.552". Everything else is ignored.
    for line in programOutput.splitlines():
        parts = line.strip().split(',')
        if len(parts) == 3 and parts[0] == 'RESULT':
            resultsForThisSize[parts[1]] = float(parts[2])


def main():
    fullMetadata = readMetadata('data/metadata.txt')
    fullWidth = int(fullMetadata['width'])
    fullHeight = int(fullMetadata['height'])
    fullBayer = numpy.fromfile('data/bayer.bin', dtype='<u2').reshape(fullHeight, fullWidth)

    allResults = []
    for cropSize in cropSizes:
        cropWidth = cropSize[0]
        cropHeight = cropSize[1]
        print('Running', cropWidth, 'x', cropHeight, '...')
        sizeFolder = writeCrop(fullBayer, fullMetadata, cropWidth, cropHeight)

        resultsForThisSize = {}
        resultsForThisSize['width'] = cropWidth
        resultsForThisSize['height'] = cropHeight
        resultsForThisSize['megapixels'] = cropWidth * cropHeight / 1000000.0

        cpuOutput = runProgram('/content/cpu_benchmark', sizeFolder)
        parseResults(cpuOutput, resultsForThisSize)
        gpuOutput = runProgram('/content/gpu_benchmark', sizeFolder)
        parseResults(gpuOutput, resultsForThisSize)

        compareCommand = ['/content/compare_outputs', 'output/cpu_output.bin', 'output/gpu_fused_output.bin']
        compareRun = subprocess.run(compareCommand, cwd=sizeFolder, capture_output=True, text=True)
        if 'RESULT: PASS' in compareRun.stdout:
            resultsForThisSize['fused_check'] = 'PASS'
        else:
            resultsForThisSize['fused_check'] = 'FAIL'

        allResults.append(resultsForThisSize)

    saveCsv(allResults)
    printTable(allResults)
    drawChart(allResults)


def saveCsv(allResults):
    columnNames = ['width', 'height', 'megapixels'] + versionNames + ['copy_in', 'copy_out', 'fused_check']
    csvFile = open('benchmark/results.csv', 'w')
    csvFile.write(','.join(columnNames) + '\n')
    for resultsForThisSize in allResults:
        rowValues = []
        for columnName in columnNames:
            rowValues.append(str(resultsForThisSize[columnName]))
        csvFile.write(','.join(rowValues) + '\n')
    csvFile.close()
    print('Saved benchmark/results.csv')


def printTable(allResults):
    print()
    print('Times in milliseconds (GPU columns = kernels only; copies listed separately)')
    header = '{:>11} {:>6} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>6}'
    print(header.format('size', 'MP', 'cpu', 'naive', 'planar', 'shared', 'fused',
                        'copy in', 'copy out', 'cpu/fused', 'check'))
    row = '{:>11} {:>6.2f} {:>9.2f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>8.0f}x {:>6}'
    for resultsForThisSize in allResults:
        sizeLabel = str(resultsForThisSize['width']) + 'x' + str(resultsForThisSize['height'])
        speedUp = resultsForThisSize['cpu'] / resultsForThisSize['fused']
        print(row.format(sizeLabel, resultsForThisSize['megapixels'],
                         resultsForThisSize['cpu'], resultsForThisSize['naive'],
                         resultsForThisSize['planar'], resultsForThisSize['shared'],
                         resultsForThisSize['fused'], resultsForThisSize['copy_in'],
                         resultsForThisSize['copy_out'], speedUp,
                         resultsForThisSize['fused_check']))


def drawChart(allResults):
    # One colour AND one marker shape per version, so the lines can be told
    # apart even without colour (colour-blind readers, black-and-white prints).
    versionColours = {'cpu': '#2a78d6', 'naive': '#eb6834', 'planar': '#1baf7a',
                      'shared': '#eda100', 'fused': '#e87ba4'}
    versionMarkers = {'cpu': 'o', 'naive': 's', 'planar': '^', 'shared': 'D', 'fused': 'v'}
    versionLabels = {'cpu': 'CPU (C++)', 'naive': 'GPU naive (M2)', 'planar': 'GPU planar 32x8 (M5)',
                     'shared': 'GPU shared denoise (M6)', 'fused': 'GPU fused (M7)'}

    megapixelValues = []
    for resultsForThisSize in allResults:
        megapixelValues.append(resultsForThisSize['megapixels'])

    figure, (timeAxes, throughputAxes) = pyplot.subplots(1, 2, figsize=(13, 5))

    # Left: time vs image size. Both axes logarithmic, because the CPU is
    # hundreds of times slower and sizes span 0.3 to 22 megapixels.
    for versionName in versionNames:
        timeValues = []
        for resultsForThisSize in allResults:
            timeValues.append(resultsForThisSize[versionName])
        timeAxes.plot(megapixelValues, timeValues, color=versionColours[versionName],
                      marker=versionMarkers[versionName], markersize=8, linewidth=2,
                      label=versionLabels[versionName])
    timeAxes.set_xscale('log')
    timeAxes.set_yscale('log')
    timeAxes.set_xlabel('Image size (megapixels, log scale)')
    timeAxes.set_ylabel('Time per image (ms, log scale)')
    timeAxes.set_title('Processing time vs image size')
    timeAxes.grid(True, which='major', color='#e4e3df', linewidth=0.8)
    timeAxes.legend(frameon=False)

    # Right: throughput (megapixels processed per millisecond) for the GPU
    # versions. A flat line means the GPU is fully busy; a rising line means
    # small images do not give it enough work.
    for versionName in versionNames:
        if versionName == 'cpu':
            continue
        throughputValues = []
        for resultsForThisSize in allResults:
            throughputValues.append(resultsForThisSize['megapixels'] / resultsForThisSize[versionName])
        throughputAxes.plot(megapixelValues, throughputValues, color=versionColours[versionName],
                            marker=versionMarkers[versionName], markersize=8, linewidth=2,
                            label=versionLabels[versionName])
    throughputAxes.set_xscale('log')
    throughputAxes.set_ylim(bottom=0)
    throughputAxes.set_xlabel('Image size (megapixels, log scale)')
    throughputAxes.set_ylabel('Megapixels per millisecond (kernels only)')
    throughputAxes.set_title('GPU throughput vs image size')
    throughputAxes.grid(True, which='major', color='#e4e3df', linewidth=0.8)
    throughputAxes.legend(frameon=False)

    # Put a tick at each tested size, labelled in plain megapixels (0.3, 1.1 ...)
    # instead of the default powers of ten, which are hard to read.
    tickLabels = []
    for megapixels in megapixelValues:
        tickLabels.append('{:.1f}'.format(megapixels))

    for axes in [timeAxes, throughputAxes]:
        axes.set_xticks(megapixelValues)
        axes.set_xticklabels(tickLabels)
        axes.minorticks_off()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

    figure.tight_layout()
    figure.savefig('benchmark/results_chart.png', dpi=130)
    print('Saved benchmark/results_chart.png')


main()
