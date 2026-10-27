# roofline_analysis.py
# Milestone 9: roofline analysis.
# Runs /content/gpu_roofline (which prints the measured times), combines them
# with the hand-counted bytes and operations per pixel below, prints a table,
# and draws a roofline chart.

import subprocess
import matplotlib.pyplot as pyplot

# ---------------------------------------------------------------------------
# T4 peak numbers, from NVIDIA's official pages (see the notebook text):
#   FP32 peak: 8.1 TFLOPS = 8100 GFLOP/s (datasheet and product page agree)
#   Memory bandwidth: product page says "over 320 GB/s", datasheet says 300 GB/s.
#   We use 320 for the roofline and ALSO report the bandwidth we measure.
# ---------------------------------------------------------------------------
peakGigaflops = 8100.0
specBandwidthGigabytes = 320.0
ridgePoint = peakGigaflops / specBandwidthGigabytes   # FLOP per byte where the roofs meet

# ---------------------------------------------------------------------------
# Hand counts PER PIXEL. "Bytes" = the minimum that must cross global memory
# if every value is read once and written once (caches may cause extra traffic
# in practice; we count the unavoidable part). "Operations" = floating-point
# operations: add, subtract, multiply, divide, compare-and-clamp each count 1;
# a multiply-add (FMA) counts 2; powf is counted as 1 even though it is really
# about ten instructions (so gamma's count is an UNDER-estimate).
#
# Stage      read  write  operations    where the operations come from
# black        2     4        4         subtract, divide, clamp low, clamp high
# demosaic     4    12       12         4 averages: 3+1, 3+1, 1+1, 1+1 adds/divides
# white       12    12        6         3 multiplies + 3 clamps
# denoise     12    12       57         per colour 9 FMAs (18) + 1 divide = 19, x3
# gamma       12     3       15         per colour 2 clamps + powf + multiply + add = 5, x3
# 5 kernels   42    43       94         total: 85 bytes
#
# Fused kernel: reads only the raw uint16 (2 bytes) and writes 3 bytes = 5 bytes.
# But it repeats some work for the tile halos:
#   black level for 36x12 = 432 values per 256 pixels -> 4 x 432/256 = 6.75
#   demosaic + white balance for 34x10 = 340 per 256  -> 18 x 340/256 = 23.91
#   denoise 57 + gamma 15                              -> 72
#   total about 102.66 operations per pixel.
# ---------------------------------------------------------------------------
stageCounts = {
    'black':    {'bytes': 6.0,  'operations': 4.0},
    'demosaic': {'bytes': 16.0, 'operations': 12.0},
    'white':    {'bytes': 24.0, 'operations': 6.0},
    'denoise':  {'bytes': 24.0, 'operations': 57.0},
    'gamma':    {'bytes': 15.0, 'operations': 15.0},
}
fusedOperations = 4.0 * 432.0 / 256.0 + 18.0 * 340.0 / 256.0 + 57.0 + 15.0
fusedBytes = 5.0


def runMeasurements():
    completedRun = subprocess.run(['/content/gpu_roofline'], capture_output=True, text=True)
    if completedRun.returncode != 0:
        print('ERROR running gpu_roofline:')
        print(completedRun.stdout)
        print(completedRun.stderr)
    measurements = {}
    for line in completedRun.stdout.splitlines():
        parts = line.strip().split(',')
        if len(parts) == 3 and parts[0] == 'RESULT':
            measurements[parts[1]] = float(parts[2])
    return measurements


def analyseKernel(rowName, bytesPerPixel, operationsPerPixel, milliseconds, pixelCount, measuredBandwidth):
    seconds = milliseconds / 1000.0
    arithmeticIntensity = operationsPerPixel / bytesPerPixel
    achievedGigabytes = bytesPerPixel * pixelCount / seconds / 1.0e9
    achievedGigaflops = operationsPerPixel * pixelCount / seconds / 1.0e9

    # The roofline: the best a kernel with this intensity could possibly do
    # is the LOWER of the two roofs.
    memoryRoof = arithmeticIntensity * specBandwidthGigabytes
    attainableGigaflops = min(peakGigaflops, memoryRoof)
    if arithmeticIntensity < ridgePoint:
        predictedLimit = 'memory'
    else:
        predictedLimit = 'compute'

    analysis = {
        'name': rowName,
        'milliseconds': milliseconds,
        'intensity': arithmeticIntensity,
        'gigabytes': achievedGigabytes,
        'percentSpecBandwidth': 100.0 * achievedGigabytes / specBandwidthGigabytes,
        'percentMeasuredBandwidth': 100.0 * achievedGigabytes / measuredBandwidth,
        'gigaflops': achievedGigaflops,
        'percentPeakFlops': 100.0 * achievedGigaflops / peakGigaflops,
        'percentOfRoof': 100.0 * achievedGigaflops / attainableGigaflops,
        'predictedLimit': predictedLimit,
    }
    return analysis


def buildAnalyses(measurements):
    pixelCount = measurements['pixels']
    measuredBandwidth = measurements['copy_kernel_GBps']
    analyses = []

    for prefix in ['naive', 'planar']:
        totalMilliseconds = 0.0
        for stageName in ['black', 'demosaic', 'white', 'denoise', 'gamma']:
            stageMilliseconds = measurements[prefix + '_' + stageName]
            totalMilliseconds = totalMilliseconds + stageMilliseconds
            analyses.append(analyseKernel(prefix + ' ' + stageName,
                                          stageCounts[stageName]['bytes'],
                                          stageCounts[stageName]['operations'],
                                          stageMilliseconds, pixelCount, measuredBandwidth))
        analyses.append(analyseKernel(prefix + ' ALL 5', 85.0, 94.0,
                                      totalMilliseconds, pixelCount, measuredBandwidth))

    analyses.append(analyseKernel('shared denoise', stageCounts['denoise']['bytes'],
                                  stageCounts['denoise']['operations'],
                                  measurements['shared_denoise'], pixelCount, measuredBandwidth))
    analyses.append(analyseKernel('fused', fusedBytes, fusedOperations,
                                  measurements['fused_all'], pixelCount, measuredBandwidth))
    return analyses


def printReport(measurements, analyses):
    print('T4 peaks: FP32', peakGigaflops, 'GFLOP/s | memory', specBandwidthGigabytes,
          'GB/s (spec) | ridge point', round(ridgePoint, 1), 'FLOP/byte')
    print('Measured copy-kernel bandwidth: {:.1f} GB/s ({:.0f}% of the 320 GB/s spec)'.format(
        measurements['copy_kernel_GBps'], 100.0 * measurements['copy_kernel_GBps'] / specBandwidthGigabytes))
    print()
    header = '{:<16} {:>8} {:>9} {:>8} {:>9} {:>10} {:>9} {:>9} {:>9}'
    print(header.format('kernel', 'ms', 'FLOP/B', 'GB/s', '%spec BW', '%meas. BW',
                        'GFLOP/s', '%peak FL', 'roofline'))
    row = '{:<16} {:>8.3f} {:>9.2f} {:>8.1f} {:>8.0f}% {:>9.0f}% {:>9.1f} {:>8.1f}% {:>9}'
    for analysis in analyses:
        print(row.format(analysis['name'], analysis['milliseconds'], analysis['intensity'],
                         analysis['gigabytes'], analysis['percentSpecBandwidth'],
                         analysis['percentMeasuredBandwidth'], analysis['gigaflops'],
                         analysis['percentPeakFlops'], analysis['predictedLimit']))


def drawRoofline(measurements, analyses):
    figure, axes = pyplot.subplots(figsize=(9, 6))

    # The two roofs. Left of the ridge, the sloped memory roof is lower;
    # right of it, the flat compute roof is lower.
    intensityValues = []
    intensity = 0.05
    while intensity <= 200.0:
        intensityValues.append(intensity)
        intensity = intensity * 1.1
    roofValues = []
    measuredRoofValues = []
    for intensity in intensityValues:
        roofValues.append(min(peakGigaflops, intensity * specBandwidthGigabytes))
        measuredRoofValues.append(min(peakGigaflops, intensity * measurements['copy_kernel_GBps']))
    axes.plot(intensityValues, roofValues, color='#57564f', linewidth=2,
              label='T4 roof (320 GB/s spec, 8.1 TFLOP/s)')
    axes.plot(intensityValues, measuredRoofValues, color='#57564f', linewidth=1.5, linestyle='--',
              label='Measured copy bandwidth roof')

    # One colour and marker per version (validated categorical palette, Milestone 8).
    groupStyles = {
        'naive':  {'colour': '#eb6834', 'marker': 's', 'label': 'naive stages (M2)'},
        'planar': {'colour': '#1baf7a', 'marker': '^', 'label': 'planar stages (M5)'},
        'shared': {'colour': '#eda100', 'marker': 'D', 'label': 'shared denoise (M6)'},
        'fused':  {'colour': '#e87ba4', 'marker': 'v', 'label': 'fused (M7)'},
    }
    labelledGroups = []
    for analysis in analyses:
        groupName = analysis['name'].split(' ')[0]
        style = groupStyles[groupName]
        legendLabel = None
        if groupName not in labelledGroups:
            legendLabel = style['label']
            labelledGroups.append(groupName)
        axes.scatter(analysis['intensity'], analysis['gigaflops'], color=style['colour'],
                     marker=style['marker'], s=70, edgecolors='white', linewidths=1.5,
                     zorder=3, label=legendLabel)
        # Label only the planar stages and the fused kernel, to avoid clutter.
        if groupName == 'planar' or groupName == 'fused':
            axes.annotate(analysis['name'], (analysis['intensity'], analysis['gigaflops']),
                          textcoords='offset points', xytext=(7, -3), fontsize=8, color='#57564f')

    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel('Arithmetic intensity (FLOP per byte of global memory traffic)')
    axes.set_ylabel('Achieved performance (GFLOP/s)')
    axes.set_title('Roofline: T4, full image (' + str(int(measurements['pixels'])) + ' pixels)')
    axes.grid(True, which='major', color='#e4e3df', linewidth=0.8)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.legend(frameon=False, fontsize=9, loc='lower right')
    figure.tight_layout()
    figure.savefig('output/roofline_chart.png', dpi=130)
    print()
    print('Saved output/roofline_chart.png')


def main():
    measurements = runMeasurements()
    analyses = buildAnalyses(measurements)
    printReport(measurements, analyses)
    drawRoofline(measurements, analyses)


main()
