#include "spbitnet/cuda_utils.h"
#include "spbitnet/inference.h"
#include "spbitnet/model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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

        // Create inference engine (allocates KV cache + scratch)
        spbitnet::InferenceEngine engine(model);
        spbitnet::print_vram_usage("after engine init");

        const auto& cfg = model.config();

        // --- Greedy text generation (token IDs only — no tokenizer yet) ---
        printf("\n--- Generation (greedy, token IDs) ---\n");
        printf("Prompt: \"%s\" (NOTE: no tokenizer yet, using BOS token)\n", prompt.c_str());
        printf("Max tokens: %d\n\n", max_tokens);

        std::vector<float> h_logits(cfg.vocab_size);
        std::vector<int> generated;

        // Start with BOS token
        int token = cfg.bos_token_id;
        generated.push_back(token);
        printf("token[0] = %d (BOS)\n", token);

        for (int t = 0; t < max_tokens; ++t) {
            const float* d_logits = engine.forward(token);

            // Copy logits to host
            CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                                  cfg.vocab_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Greedy: argmax
            token = static_cast<int>(
                std::max_element(h_logits.begin(), h_logits.end())
                - h_logits.begin());

            generated.push_back(token);
            printf("token[%d] = %d", t + 1, token);

            // Show top-1 logit value for debugging
            printf("  (logit=%.2f)", h_logits[token]);
            printf("\n");

            if (token == cfg.eos_token_id) {
                printf("(EOS)\n");
                break;
            }
        }

        printf("\nGenerated %d tokens total.\n", static_cast<int>(generated.size()));
        spbitnet::print_vram_usage("after generation");

    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
