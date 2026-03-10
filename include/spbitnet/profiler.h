#pragma once

/// \file profiler.h
/// \brief Lightweight per-kernel CUDA event profiler for inference timing.

#include "spbitnet/cuda_utils.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace spbitnet {

/// Accumulates per-kernel timing via CUDA events.
/// When disabled (default), all methods are no-ops with zero overhead.
class Profiler {
public:
    Profiler() = default;
    ~Profiler() { destroy_events(); }

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    void set_enabled(bool enabled) {
        if (enabled && !enabled_) {
            CUDA_CHECK(cudaEventCreate(&ev_start_));
            CUDA_CHECK(cudaEventCreate(&ev_stop_));
        } else if (!enabled && enabled_) {
            destroy_events();
        }
        enabled_ = enabled;
    }

    bool enabled() const { return enabled_; }

    /// Call before a kernel launch.
    void begin(const char* name, cudaStream_t stream = nullptr) {
        if (!enabled_) return;
        current_name_ = name;
        CUDA_CHECK(cudaEventRecord(ev_start_, stream));
    }

    /// Call after a kernel launch (or cudaMemcpy).
    void end(cudaStream_t stream = nullptr) {
        if (!enabled_) return;
        CUDA_CHECK(cudaEventRecord(ev_stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start_, ev_stop_));
        records_[current_name_].push_back(ms * 1000.0f);  // store as microseconds
    }

    /// Reset all accumulated timings.
    void reset() { records_.clear(); }

    /// Print summary table sorted by total time descending.
    void report() const {
        if (records_.empty()) return;

        struct Row {
            std::string name;
            int calls;
            float total_us;
            float mean_us;
            float min_us;
            float max_us;
        };

        std::vector<Row> rows;
        float grand_total = 0.0f;

        for (auto& [name, times] : records_) {
            Row r;
            r.name = name;
            r.calls = static_cast<int>(times.size());
            r.total_us = 0.0f;
            r.min_us = times[0];
            r.max_us = times[0];
            for (float t : times) {
                r.total_us += t;
                if (t < r.min_us) r.min_us = t;
                if (t > r.max_us) r.max_us = t;
            }
            r.mean_us = r.total_us / r.calls;
            grand_total += r.total_us;
            rows.push_back(r);
        }

        std::sort(rows.begin(), rows.end(),
                  [](const Row& a, const Row& b) { return a.total_us > b.total_us; });

        printf("\n%-28s %6s %10s %10s %10s %10s %6s\n",
               "Kernel", "Calls", "Total(us)", "Mean(us)", "Min(us)", "Max(us)", "  %");
        printf("%-28s %6s %10s %10s %10s %10s %6s\n",
               "----------------------------", "------", "----------",
               "----------", "----------", "----------", "------");

        for (auto& r : rows) {
            float pct = grand_total > 0.0f ? 100.0f * r.total_us / grand_total : 0.0f;
            printf("%-28s %6d %10.1f %10.1f %10.1f %10.1f %5.1f%%\n",
                   r.name.c_str(), r.calls, r.total_us, r.mean_us, r.min_us, r.max_us, pct);
        }

        printf("%-28s %6s %10.1f\n", "TOTAL", "", grand_total);
    }

private:
    bool enabled_ = false;
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_stop_ = nullptr;
    std::string current_name_;
    std::unordered_map<std::string, std::vector<float>> records_;

    void destroy_events() {
        if (ev_start_) { cudaEventDestroy(ev_start_); ev_start_ = nullptr; }
        if (ev_stop_)  { cudaEventDestroy(ev_stop_);  ev_stop_  = nullptr; }
    }
};

}  // namespace spbitnet
