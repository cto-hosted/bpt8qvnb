#pragma once

#include <vector>

namespace dsp {

struct SimilarityResult {
    int index_a{-1};
    int index_b{-1};
    double score{0.0};
    bool is_duplicate{false};
};

class SimilarityDetector {
public:
    explicit SimilarityDetector(int sample_rate = 44100, double threshold = 0.85);

    std::vector<double> compute_features(const std::vector<float>& audio, int channels) const;
    double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b) const;
    bool are_similar(const std::vector<float>& audio_a, const std::vector<float>& audio_b, int channels) const;

    std::vector<SimilarityResult> find_duplicates(
        const std::vector<std::vector<float>>& clips,
        int channels) const;

private:
    std::vector<float> to_mono(const std::vector<float>& audio, int channels) const;
    std::vector<double> compute_rms_frames(const std::vector<float>& mono) const;
    std::vector<double> compute_spectral_centroid(const std::vector<float>& mono) const;
    double compute_zero_crossing_rate(const std::vector<float>& mono) const;

    int sample_rate_;
    double threshold_;
    int hop_length_;
    static constexpr int kFftSize = 2048;
};

} // namespace dsp
