#include "loop_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dsp {

LoopExtractor::LoopExtractor(const LoopExtractorConfig& config) : cfg_(config) {}

int LoopExtractor::target_samples() const {
    double beat_duration = 60.0 / cfg_.bpm;
    double bar_duration = beat_duration * cfg_.beats_per_bar;
    return static_cast<int>(cfg_.bars * bar_duration * cfg_.sample_rate);
}

std::vector<float> LoopExtractor::to_mono(const std::vector<float>& audio, int channels) const {
    if (channels == 1) return audio;
    size_t n_frames = audio.size() / channels;
    std::vector<float> mono(n_frames, 0.0f);
    for (size_t i = 0; i < n_frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += audio[i * channels + c];
        }
        mono[i] = sum / channels;
    }
    return mono;
}

std::vector<float> LoopExtractor::compute_onset_envelope(const std::vector<float>& mono) const {
    int hop = std::max(1, static_cast<int>(cfg_.sample_rate * 0.01));
    size_t n_frames = mono.size() / hop;
    std::vector<float> envelope(n_frames, 0.0f);
    for (size_t i = 0; i < n_frames; ++i) {
        float sum = 0.0f;
        size_t start = i * hop;
        size_t end = std::min(start + hop, mono.size());
        for (size_t j = start; j < end; ++j) {
            sum += std::abs(mono[j]);
        }
        envelope[i] = sum / (end - start);
    }
    std::vector<float> diff(n_frames, 0.0f);
    for (size_t i = 1; i < n_frames; ++i) {
        diff[i] = std::max(0.0f, envelope[i] - envelope[i - 1]);
    }
    return diff;
}

std::vector<int64_t> LoopExtractor::pick_peaks(
    const std::vector<float>& env, int min_distance) const {
    std::vector<int64_t> peaks;
    int last = -min_distance;
    for (size_t i = 1; i + 1 < env.size(); ++i) {
        if (env[i] > env[i - 1] && env[i] >= env[i + 1]) {
            if (static_cast<int>(i) - last >= min_distance) {
                peaks.push_back(static_cast<int64_t>(i));
                last = static_cast<int>(i);
            }
        }
    }
    return peaks;
}

std::vector<int64_t> LoopExtractor::detect_beats(const std::vector<float>& mono) const {
    auto onset_env = compute_onset_envelope(mono);
    int min_dist = std::max(1, static_cast<int>(cfg_.sample_rate * 0.2 / (cfg_.sample_rate * 0.01)));
    return pick_peaks(onset_env, min_dist);
}

std::vector<float> LoopExtractor::extract(
    const std::vector<float>& audio, int channels) const {
    int target = target_samples() * channels;
    if (static_cast<int>(audio.size()) <= target) {
        if (cfg_.beat_quantize && static_cast<int>(audio.size()) < target) {
            std::vector<float> padded = audio;
            padded.resize(target, 0.0f);
            return padded;
        }
        return audio;
    }
    auto mono = to_mono(audio, channels);
    auto beats = detect_beats(mono);
    int start_frame = 0;
    if (!beats.empty()) {
        start_frame = static_cast<int>(beats[0]);
    }
    int start_sample = start_frame * channels;
    int end_sample = start_sample + target;
    if (end_sample > static_cast<int>(audio.size())) {
        start_sample = 0;
        end_sample = target;
    }
    return std::vector<float>(audio.begin() + start_sample, audio.begin() + end_sample);
}

} // namespace dsp
