#include "spbitnet/cuda_utils.h"
#include <cstdio>
#include <stdexcept>

int main(int /*argc*/, char* /*argv*/[]) {
    printf("spbitnet — Sparse-BitNet Inference on Consumer GPUs\n");
    printf("===================================================\n\n");

    try {
        spbitnet::print_device_info();
        spbitnet::print_vram_usage("startup");
    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
