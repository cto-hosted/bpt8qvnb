#pragma once

#include "crossfade_processor.hpp"
#include "loop_extractor.hpp"
#include "similarity_detector.hpp"

#include <string>
#include <vector>

namespace dsp {

struct EngineConfig {
    int sample_rate{44100};
    int channels{2};
    double bpm{120.0};
    int beats_per_bar{4};
    int bars{4};
    double transient_threshold{0.3};
    double similarity_threshold{0.85};
    double crossfade_ms{50.0};
    std::string fade_shape{"equal_power"};
    std::string mode{"rhythmic"};
};

struct ProcessResult {
    std::vector<float> audio;
    std::vector<int64_t> beat_positions;
    double similarity_score{0.0};
    bool is_duplicate{false};
    std::string error_message;
    bool success{true};
};

class DspEngine {
public:
    explicit DspEngine(const EngineConfig& config);

    ProcessResult process(const std::vector<float>& input_audio) const;

    ProcessResult process_and_crossfade(
        const std::vector<float>& audio_a,
        const std::vector<float>& audio_b) const;

    SimilarityResult check_similarity(
        const std::vector<float>& audio_a,
        const std::vector<float>& audio_b) const;

    std::vector<float> apply_crossfade(
        const std::vector<float>& audio_a,
        const std::vector<float>& audio_b) const;

    const EngineConfig& config() const { return config_; }

private:
    EngineConfig config_;
    LoopExtractor extractor_;
    SimilarityDetector similarity_;
    CrossfadeProcessor crossfader_;
};

} // namespace dsp
