// cuda_helpers.cu

#include "cuda_helpers.h"

#include <cstdlib>
#include <iostream>

void checkCudaError(cudaError_t result, const char* whatWeWereDoing) {
    if (result != cudaSuccess) {
        // cudaGetErrorString turns the error code into readable text,
        // for example "out of memory" or "invalid configuration argument".
        std::cout << "CUDA ERROR while " << whatWeWereDoing << ": "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(1);
    }
}
