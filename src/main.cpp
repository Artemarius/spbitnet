#include "spbitnet/cuda_utils.h"
#include "spbitnet/inference.h"
#include "spbitnet/model.h"
#include "spbitnet/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
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

        // Try to load tokenizer
        std::unique_ptr<spbitnet::Tokenizer> tokenizer;
        {
            std::string tok_path = model_dir + "/tokenizer.json";
            try {
                tokenizer = std::make_unique<spbitnet::Tokenizer>(
                    spbitnet::Tokenizer::load(tok_path));
                printf("Tokenizer loaded (%d tokens)\n", tokenizer->vocab_size());
            } catch (const std::exception& e) {
                printf("Warning: tokenizer not available (%s)\n", e.what());
            }
        }

        // Tokenize prompt
        std::vector<int> prompt_ids;
        if (tokenizer) {
            prompt_ids = tokenizer->encode(prompt);
            printf("Prompt: \"%s\" -> %d tokens [", prompt.c_str(),
                   static_cast<int>(prompt_ids.size()));
            for (size_t i = 0; i < prompt_ids.size(); ++i)
                printf("%s%d", i ? ", " : "", prompt_ids[i]);
            printf("]\n");
        }

        printf("Max tokens: %d\n\n", max_tokens);

        std::vector<float> h_logits(cfg.vocab_size);

        // Build input sequence: BOS + prompt tokens
        std::vector<int> input_ids;
        input_ids.push_back(cfg.bos_token_id);
        input_ids.insert(input_ids.end(), prompt_ids.begin(), prompt_ids.end());

        // Prefill: process all input tokens except the last
        for (size_t i = 0; i + 1 < input_ids.size(); ++i)
            engine.forward(input_ids[i]);

        // Last input token starts generation
        int token = input_ids.back();

        printf("--- Generation ---\n");
        if (tokenizer)
            printf("%s", prompt.c_str());

        std::vector<int> generated;
        for (int t = 0; t < max_tokens; ++t) {
            const float* d_logits = engine.forward(token);

            CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                                  cfg.vocab_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Greedy: argmax
            token = static_cast<int>(
                std::max_element(h_logits.begin(), h_logits.end())
                - h_logits.begin());

            generated.push_back(token);

            if (tokenizer) {
                printf("%s", tokenizer->decode_token(token).c_str());
                fflush(stdout);
            } else {
                printf("token[%d] = %d  (logit=%.2f)\n",
                       t, token, h_logits[token]);
            }

            if (token == cfg.eos_token_id) break;
        }

        if (tokenizer)
            printf("\n");
        printf("\n--- %d tokens generated ---\n",
               static_cast<int>(generated.size()));
        spbitnet::print_vram_usage("after generation");

    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
