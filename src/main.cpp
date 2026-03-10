#include "spbitnet/cuda_utils.h"
#include "spbitnet/inference.h"
#include "spbitnet/model.h"
#include "spbitnet/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void print_usage(const char* argv0) {
    printf("Usage: %s [OPTIONS]\n", argv0);
    printf("\nOptions:\n");
    printf("  --model <dir>       Load model from directory (convert_model.py output)\n");
    printf("  --prompt <text>     Prompt text for generation (default: \"Hello\")\n");
    printf("  --max-tokens <n>    Maximum tokens to generate (default: 32)\n");
    printf("  --dump-tokens <f>   Dump generated token IDs to JSON file\n");
    printf("  --benchmark [n]     Benchmark mode: generate n tokens (default: 128),\n");
    printf("                      report tokens/sec with warmup run\n");
    printf("  --profile           Enable per-kernel timing breakdown\n");
    printf("  --help              Show this help\n");
}

int main(int argc, char* argv[]) {
    printf("spbitnet — Sparse-BitNet Inference on Consumer GPUs\n");
    printf("===================================================\n\n");

    // Parse args
    std::string model_dir;
    std::string prompt = "Hello";
    std::string dump_tokens_path;
    int max_tokens = 32;
    bool benchmark_mode = false;
    int benchmark_tokens = 128;
    bool profile_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--dump-tokens") == 0 && i + 1 < argc) {
            dump_tokens_path = argv[++i];
        } else if (std::strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                benchmark_tokens = std::atoi(argv[++i]);
                if (benchmark_tokens <= 0) benchmark_tokens = 128;
            }
        } else if (std::strcmp(argv[i], "--profile") == 0) {
            profile_mode = true;
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

        if (profile_mode)
            engine.profiler().set_enabled(true);

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

        std::vector<float> h_logits(cfg.vocab_size);

        // Build input sequence: BOS + prompt tokens
        std::vector<int> input_ids;
        input_ids.push_back(cfg.bos_token_id);
        input_ids.insert(input_ids.end(), prompt_ids.begin(), prompt_ids.end());

        // ============================================================
        // Benchmark mode
        // ============================================================
        if (benchmark_mode) {
            const int gen_tokens = benchmark_tokens;
            printf("\n=== Benchmark Mode ===\n");
            printf("Generating %d tokens per run\n\n", gen_tokens);

            // Helper: run one full generation pass, return generated token IDs
            auto run_generation = [&](bool silent) -> std::vector<int> {
                engine.reset();

                // Prefill
                for (size_t i = 0; i + 1 < input_ids.size(); ++i)
                    engine.forward(input_ids[i]);

                int tok = input_ids.back();
                std::vector<int> gen;
                gen.reserve(gen_tokens);

                for (int t = 0; t < gen_tokens; ++t) {
                    const float* d_logits = engine.forward(tok);
                    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                                          cfg.vocab_size * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                    tok = static_cast<int>(
                        std::max_element(h_logits.begin(), h_logits.end())
                        - h_logits.begin());
                    gen.push_back(tok);
                    if (!silent && tokenizer) {
                        printf("%s", tokenizer->decode_token(tok).c_str());
                        fflush(stdout);
                    }
                    if (tok == cfg.eos_token_id) break;
                }
                return gen;
            };

            // Warmup run (primes GPU caches, JIT, etc.)
            printf("Warmup run...\n");
            run_generation(true);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Timed runs
            const int num_runs = 3;
            std::vector<double> prefill_ms_runs;
            std::vector<double> decode_ms_runs;
            std::vector<int>    decode_count_runs;

            printf("Running %d timed iterations...\n\n", num_runs);

            for (int r = 0; r < num_runs; ++r) {
                engine.reset();
                // Reset profiler so report only reflects the last run
                if (profile_mode)
                    engine.profiler().reset();
                CUDA_CHECK(cudaDeviceSynchronize());

                // Timed prefill
                auto t0 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i + 1 < input_ids.size(); ++i)
                    engine.forward(input_ids[i]);
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t1 = std::chrono::high_resolution_clock::now();

                // Timed decode
                int tok = input_ids.back();
                int decode_count = 0;

                auto t2 = std::chrono::high_resolution_clock::now();
                for (int t = 0; t < gen_tokens; ++t) {
                    const float* d_logits = engine.forward(tok);
                    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits,
                                          cfg.vocab_size * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                    tok = static_cast<int>(
                        std::max_element(h_logits.begin(), h_logits.end())
                        - h_logits.begin());
                    ++decode_count;
                    if (tok == cfg.eos_token_id) break;
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t3 = std::chrono::high_resolution_clock::now();

                double pf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                double dc_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

                prefill_ms_runs.push_back(pf_ms);
                decode_ms_runs.push_back(dc_ms);
                decode_count_runs.push_back(decode_count);

                printf("  Run %d: prefill %.1f ms (%d tokens), decode %.1f ms (%d tokens, %.1f tok/s)\n",
                       r + 1, pf_ms,
                       static_cast<int>(input_ids.size()) - 1,
                       dc_ms, decode_count,
                       decode_count / (dc_ms / 1000.0));
            }

            // Summary statistics
            auto median = [](std::vector<double> v) -> double {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };

            int prefill_tokens = static_cast<int>(input_ids.size()) - 1;
            double pf_median = median(prefill_ms_runs);
            double dc_median = median(decode_ms_runs);
            int dc_count = decode_count_runs[0];  // same prompt → same EOS behavior

            printf("\n=== Benchmark Results ===\n");
            printf("Model: %s\n", model_dir.c_str());
            printf("Prompt tokens: %d, Generated tokens: %d\n",
                   prefill_tokens, dc_count);
            printf("Prefill:  %.1f ms (median), %.1f tok/s\n",
                   pf_median,
                   prefill_tokens > 0 ? prefill_tokens / (pf_median / 1000.0) : 0.0);
            printf("Decode:   %.1f ms (median), %.2f ms/tok, %.1f tok/s\n",
                   dc_median,
                   dc_median / dc_count,
                   dc_count / (dc_median / 1000.0));
            printf("Total:    %.1f ms for %d tokens\n",
                   pf_median + dc_median,
                   prefill_tokens + dc_count);

            spbitnet::print_vram_usage("peak");

            // Profile report (if enabled)
            if (profile_mode) {
                printf("\nNote: --profile adds ~2-3x overhead (per-kernel sync).\n");
                printf("Tok/s numbers above reflect profiled speed, not peak.\n");
                printf("\n=== Per-Kernel Profile (last timed run) ===\n");
                engine.profiler().report();
            }

            return 0;
        }

        // ============================================================
        // Normal generation mode
        // ============================================================
        printf("Max tokens: %d\n\n", max_tokens);

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

        // Profile report (if enabled in normal mode)
        if (profile_mode) {
            printf("\n=== Per-Kernel Profile ===\n");
            engine.profiler().report();
        }

        // Dump tokens as JSON for validation against PyTorch reference
        if (!dump_tokens_path.empty()) {
            std::ofstream ofs(dump_tokens_path);
            if (!ofs)
                throw std::runtime_error("Cannot open " + dump_tokens_path);

            ofs << "{\n";
            ofs << "  \"prompt\": \"" << prompt << "\",\n";

            ofs << "  \"prompt_ids\": [";
            for (size_t i = 0; i < prompt_ids.size(); ++i)
                ofs << (i ? ", " : "") << prompt_ids[i];
            ofs << "],\n";

            ofs << "  \"input_ids\": [";
            for (size_t i = 0; i < input_ids.size(); ++i)
                ofs << (i ? ", " : "") << input_ids[i];
            ofs << "],\n";

            ofs << "  \"generated_ids\": [";
            for (size_t i = 0; i < generated.size(); ++i)
                ofs << (i ? ", " : "") << generated[i];
            ofs << "]\n";

            ofs << "}\n";
            printf("Token IDs saved to %s\n", dump_tokens_path.c_str());
        }

    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
