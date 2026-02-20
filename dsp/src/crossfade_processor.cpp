#include "crossfade_processor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dsp {

FadeShape fade_shape_from_string(const std::string& name) {
    if (name == "linear") return FadeShape::Linear;
    if (name == "equal_power") return FadeShape::EqualPower;
    if (name == "logarithmic") return FadeShape::Logarithmic;
    throw std::invalid_argument("Unknown fade shape: " + name);
}

CrossfadeProcessor::CrossfadeProcessor(int sample_rate, double duration_ms)
    : sample_rate_(sample_rate), duration_ms_(duration_ms) {}

int CrossfadeProcessor::duration_samples() const {
    return std::max(1, static_cast<int>(duration_ms_ * sample_rate_ / 1000.0));
}

void CrossfadeProcessor::build_fade_curves(
    int n_samples,
    FadeShape shape,
    std::vector<float>& fade_out_curve,
    std::vector<float>& fade_in_curve) const {
    fade_out_curve.resize(n_samples);
    fade_in_curve.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float t = static_cast<float>(i) / (n_samples - 1);
        switch (shape) {
            case FadeShape::Linear:
                fade_out_curve[i] = 1.0f - t;
                fade_in_curve[i] = t;
                break;
            case FadeShape::EqualPower:
                fade_out_curve[i] = std::cos(t * static_cast<float>(M_PI) / 2.0f);
                fade_in_curve[i] = std::sin(t * static_cast<float>(M_PI) / 2.0f);
                break;
            case FadeShape::Logarithmic:
                fade_out_curve[i] = std::sqrt(1.0f - t);
                fade_in_curve[i] = std::sqrt(t);
                break;
        }
    }
}

std::vector<float> CrossfadeProcessor::crossfade(
    const std::vector<float>& audio_a,
    const std::vector<float>& audio_b,
    int channels,
    FadeShape shape) const {
    int n_fade = std::min({duration_samples(),
                           static_cast<int>(audio_a.size() / channels),
                           static_cast<int>(audio_b.size() / channels)});
    n_fade = std::max(n_fade, 1);
    int fade_samples = n_fade * channels;

    std::vector<float> fade_out_curve, fade_in_curve;
    build_fade_curves(n_fade, shape, fade_out_curve, fade_in_curve);

    size_t body_a_size = audio_a.size() - fade_samples;
    size_t body_b_size = audio_b.size() - fade_samples;
    std::vector<float> result;
    result.reserve(body_a_size + fade_samples + body_b_size);
    result.insert(result.end(), audio_a.begin(), audio_a.begin() + body_a_size);
    for (int i = 0; i < n_fade; ++i) {
        for (int c = 0; c < channels; ++c) {
            size_t idx_a = body_a_size + i * channels + c;
            size_t idx_b = static_cast<size_t>(i * channels + c);
            float sample_a = (idx_a < audio_a.size()) ? audio_a[idx_a] : 0.0f;
            float sample_b = (idx_b < audio_b.size()) ? audio_b[idx_b] : 0.0f;
            result.push_back(sample_a * fade_out_curve[i] + sample_b * fade_in_curve[i]);
        }
    }
    result.insert(result.end(), audio_b.begin() + fade_samples, audio_b.end());
    return result;
}

std::vector<float> CrossfadeProcessor::fade_in(
    const std::vector<float>& audio,
    int channels,
    FadeShape shape) const {
    int n_fade = std::min(duration_samples(), static_cast<int>(audio.size() / channels));
    std::vector<float> fade_out_curve, fade_in_curve;
    build_fade_curves(n_fade, shape, fade_out_curve, fade_in_curve);
    std::vector<float> result = audio;
    for (int i = 0; i < n_fade; ++i) {
        for (int c = 0; c < channels; ++c) {
            result[i * channels + c] *= fade_in_curve[i];
        }
    }
    return result;
}

std::vector<float> CrossfadeProcessor::fade_out(
    const std::vector<float>& audio,
    int channels,
    FadeShape shape) const {
    int n_fade = std::min(duration_samples(), static_cast<int>(audio.size() / channels));
    std::vector<float> fade_out_curve, fade_in_curve;
    build_fade_curves(n_fade, shape, fade_out_curve, fade_in_curve);
    std::vector<float> result = audio;
    int n_frames = audio.size() / channels;
    int start_frame = n_frames - n_fade;
    for (int i = 0; i < n_fade; ++i) {
        for (int c = 0; c < channels; ++c) {
            result[(start_frame + i) * channels + c] *= fade_out_curve[i];
        }
    }
    return result;
}

std::vector<float> CrossfadeProcessor::make_seamless(
    const std::vector<float>& audio,
    int channels,
    FadeShape shape) const {
    int n_frames = audio.size() / channels;
    int n_fade = std::min(duration_samples(), n_frames / 4);
    n_fade = std::max(n_fade, 1);
    std::vector<float> fade_out_curve, fade_in_curve;
    build_fade_curves(n_fade, shape, fade_out_curve, fade_in_curve);
    std::vector<float> result = audio;
    for (int i = 0; i < n_fade; ++i) {
        for (int c = 0; c < channels; ++c) {
            result[i * channels + c] *= fade_in_curve[i];
            result[(n_frames - n_fade + i) * channels + c] *= fade_out_curve[i];
        }
    }
    return result;
}

} // namespace dsp
