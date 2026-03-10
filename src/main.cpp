#include "spbitnet/cuda_utils.h"
#include "spbitnet/model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

static void print_usage(const char* argv0) {
    printf("Usage: %s [OPTIONS]\n", argv0);
    printf("\nOptions:\n");
    printf("  --model <dir>    Load model from directory (convert_model.py output)\n");
    printf("  --prompt <text>  Prompt text for generation (default: \"Hello\")\n");
    printf("  --max-tokens <n> Maximum tokens to generate (default: 32)\n");
    printf("  --help           Show this help\n");
}

int main(int argc, char* argv[]) {
    printf("spbitnet — Sparse-BitNet Inference on Consumer GPUs\n");
    printf("===================================================\n\n");

    // Parse args
    std::string model_dir;
    std::string prompt = "Hello";
    int max_tokens = 32;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        spbitnet::print_device_info();
        spbitnet::print_vram_usage("startup");

        if (model_dir.empty()) {
            printf("No model specified. Use --model <dir> to load a model.\n");
            return 0;
        }

        // Load model weights to GPU
        auto model = spbitnet::Model::load(model_dir);
        spbitnet::print_vram_usage("after model load");

        // TODO (Phase 5): tokenize, forward pass, generate
        printf("Model loaded successfully. Inference not yet implemented.\n");
        printf("Prompt: \"%s\"\n", prompt.c_str());
        printf("Max tokens: %d\n", max_tokens);

    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
