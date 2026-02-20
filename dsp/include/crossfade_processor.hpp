#pragma once

#include <string>
#include <vector>

namespace dsp {

enum class FadeShape {
    Linear,
    EqualPower,
    Logarithmic,
};

FadeShape fade_shape_from_string(const std::string& name);

class CrossfadeProcessor {
public:
    explicit CrossfadeProcessor(int sample_rate = 44100, double duration_ms = 50.0);

    std::vector<float> crossfade(
        const std::vector<float>& audio_a,
        const std::vector<float>& audio_b,
        int channels,
        FadeShape shape = FadeShape::EqualPower) const;

    std::vector<float> fade_in(
        const std::vector<float>& audio,
        int channels,
        FadeShape shape = FadeShape::EqualPower) const;

    std::vector<float> fade_out(
        const std::vector<float>& audio,
        int channels,
        FadeShape shape = FadeShape::EqualPower) const;

    std::vector<float> make_seamless(
        const std::vector<float>& audio,
        int channels,
        FadeShape shape = FadeShape::EqualPower) const;

    int duration_samples() const;

private:
    void build_fade_curves(
        int n_samples,
        FadeShape shape,
        std::vector<float>& fade_out_curve,
        std::vector<float>& fade_in_curve) const;

    int sample_rate_;
    double duration_ms_;
};

} // namespace dsp
