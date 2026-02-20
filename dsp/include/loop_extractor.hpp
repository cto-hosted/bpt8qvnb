#pragma once

#include <cstdint>
#include <vector>

namespace dsp {

struct LoopExtractorConfig {
    double bpm{120.0};
    int beats_per_bar{4};
    int bars{4};
    double transient_threshold{0.3};
    bool beat_quantize{true};
    int sample_rate{44100};
};

class LoopExtractor {
public:
    explicit LoopExtractor(const LoopExtractorConfig& config);

    std::vector<float> extract(const std::vector<float>& audio, int channels) const;
    std::vector<int64_t> detect_beats(const std::vector<float>& mono) const;
    std::vector<float> to_mono(const std::vector<float>& audio, int channels) const;

private:
    int target_samples() const;
    std::vector<float> compute_onset_envelope(const std::vector<float>& mono) const;
    std::vector<int64_t> pick_peaks(const std::vector<float>& env, int min_distance) const;

    LoopExtractorConfig cfg_;
};

} // namespace dsp
