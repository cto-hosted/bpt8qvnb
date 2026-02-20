#include "dsp_engine.hpp"

#include <algorithm>
#include <cmath>

namespace dsp {

DspEngine::DspEngine(const EngineConfig& config)
    : config_(config),
      extractor_(LoopExtractorConfig{
          config.bpm,
          config.beats_per_bar,
          config.bars,
          config.transient_threshold,
          true,
          config.sample_rate,
      }),
      similarity_(config.sample_rate, config.similarity_threshold),
      crossfader_(config.sample_rate, config.crossfade_ms) {}

ProcessResult DspEngine::process(const std::vector<float>& input_audio) const {
    ProcessResult result;
    if (input_audio.empty()) {
        result.success = false;
        result.error_message = "Input audio is empty";
        return result;
    }
    try {
        result.audio = extractor_.extract(input_audio, config_.channels);
        auto mono = extractor_.to_mono(result.audio, config_.channels);
        result.beat_positions = extractor_.detect_beats(mono);
        auto shape = fade_shape_from_string(config_.fade_shape);
        result.audio = crossfader_.make_seamless(result.audio, config_.channels, shape);
        float peak = 0.0f;
        for (float s : result.audio) peak = std::max(peak, std::abs(s));
        if (peak > 0.95f) {
            float gain = 0.95f / peak;
            for (float& s : result.audio) s *= gain;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    return result;
}

ProcessResult DspEngine::process_and_crossfade(
    const std::vector<float>& audio_a,
    const std::vector<float>& audio_b) const {
    ProcessResult result;
    if (audio_a.empty() || audio_b.empty()) {
        result.success = false;
        result.error_message = "Input audio is empty";
        return result;
    }
    try {
        auto processed_a = extractor_.extract(audio_a, config_.channels);
        auto processed_b = extractor_.extract(audio_b, config_.channels);
        auto shape = fade_shape_from_string(config_.fade_shape);
        result.audio = crossfader_.crossfade(processed_a, processed_b, config_.channels, shape);
        auto sim = check_similarity(audio_a, audio_b);
        result.similarity_score = sim.score;
        result.is_duplicate = sim.is_duplicate;
        float peak = 0.0f;
        for (float s : result.audio) peak = std::max(peak, std::abs(s));
        if (peak > 0.95f) {
            float gain = 0.95f / peak;
            for (float& s : result.audio) s *= gain;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    return result;
}

SimilarityResult DspEngine::check_similarity(
    const std::vector<float>& audio_a,
    const std::vector<float>& audio_b) const {
    auto fa = similarity_.compute_features(audio_a, config_.channels);
    auto fb = similarity_.compute_features(audio_b, config_.channels);
    double score = similarity_.cosine_similarity(fa, fb);
    return {0, 1, score, score >= config_.similarity_threshold};
}

std::vector<float> DspEngine::apply_crossfade(
    const std::vector<float>& audio_a,
    const std::vector<float>& audio_b) const {
    auto shape = fade_shape_from_string(config_.fade_shape);
    return crossfader_.crossfade(audio_a, audio_b, config_.channels, shape);
}

} // namespace dsp
