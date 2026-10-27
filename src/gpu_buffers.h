// gpu_buffers.h
// All the GPU memory one pipeline run needs, kept together in one struct so
// we can allocate it, pass it around and free it with single function calls.
// Later milestones reuse this.

#pragma once

#include <cstddef>
#include <cstdint>

struct DeviceBuffers {
    uint16_t* deviceBayer;
    float* deviceNormalised;
    float* deviceRgb;
    float* deviceDenoised;
    unsigned char* deviceOutput;

    size_t bayerBytes;
    size_t normalisedBytes;
    size_t rgbFloatBytes;
    size_t outputBytes;
};

void allocateDeviceBuffers(DeviceBuffers& buffers, int pixelCount);
void freeDeviceBuffers(DeviceBuffers& buffers);
