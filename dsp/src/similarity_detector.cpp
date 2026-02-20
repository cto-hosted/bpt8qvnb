#include "similarity_detector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dsp {

static constexpr int kMaxCentroidFrames = 8;
static constexpr int kCentroidWindowSize = 256;

SimilarityDetector::SimilarityDetector(int sample_rate, double threshold)
    : sample_rate_(sample_rate),
      threshold_(threshold),
      hop_length_(std::max(1, static_cast<int>(sample_rate * 0.01))) {}

std::vector<float> SimilarityDetector::to_mono(
    const std::vector<float>& audio, int channels) const {
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

std::vector<double> SimilarityDetector::compute_rms_frames(
    const std::vector<float>& mono) const {
    size_t n_frames = mono.size() / hop_length_;
    std::vector<double> rms(n_frames, 0.0);
    for (size_t i = 0; i < n_frames; ++i) {
        double sum_sq = 0.0;
        size_t start = i * hop_length_;
        size_t end = std::min(start + hop_length_, mono.size());
        for (size_t j = start; j < end; ++j) {
            sum_sq += static_cast<double>(mono[j]) * mono[j];
        }
        rms[i] = std::sqrt(sum_sq / (end - start));
    }
    return rms;
}

std::vector<double> SimilarityDetector::compute_spectral_centroid(
    const std::vector<float>& mono) const {
    size_t n = std::min(static_cast<size_t>(kCentroidWindowSize), mono.size());
    int n_frames = std::min(kMaxCentroidFrames, static_cast<int>(mono.size() / n));
    if (n_frames < 1) n_frames = 1;

    std::vector<double> centroids;
    centroids.reserve(n_frames);

    size_t stride = (n_frames > 1) ? (mono.size() - n) / (n_frames - 1) : 0;
    for (int frame = 0; frame < n_frames; ++frame) {
        size_t start = std::min(frame * stride, mono.size() - n);
        double sum_weight = 0.0;
        double sum_freq = 0.0;
        for (size_t k = 0; k < n / 2; ++k) {
            double re = 0.0, im = 0.0;
            double angle_base = -2.0 * M_PI * static_cast<double>(k) / n;
            for (size_t j = 0; j < n; ++j) {
                double angle = angle_base * j;
                re += mono[start + j] * std::cos(angle);
                im += mono[start + j] * std::sin(angle);
            }
            double mag = std::sqrt(re * re + im * im);
            double freq = static_cast<double>(k) * sample_rate_ / n;
            sum_weight += mag;
            sum_freq += freq * mag;
        }
        if (sum_weight > 1e-10) {
            centroids.push_back(sum_freq / sum_weight);
        } else {
            centroids.push_back(0.0);
        }
    }
    return centroids;
}

double SimilarityDetector::compute_zero_crossing_rate(
    const std::vector<float>& mono) const {
    if (mono.size() < 2) return 0.0;
    int crossings = 0;
    for (size_t i = 1; i < mono.size(); ++i) {
        if ((mono[i] >= 0.0f) != (mono[i - 1] >= 0.0f)) {
            ++crossings;
        }
    }
    return static_cast<double>(crossings) / mono.size();
}

std::vector<double> SimilarityDetector::compute_features(
    const std::vector<float>& audio, int channels) const {
    auto mono = to_mono(audio, channels);
    auto rms = compute_rms_frames(mono);
    double mean_rms = 0.0;
    double std_rms = 0.0;
    if (!rms.empty()) {
        mean_rms = std::accumulate(rms.begin(), rms.end(), 0.0) / rms.size();
        double var = 0.0;
        for (double v : rms) var += (v - mean_rms) * (v - mean_rms);
        std_rms = std::sqrt(var / rms.size());
    }
    auto centroids = compute_spectral_centroid(mono);
    double mean_centroid = 0.0;
    double std_centroid = 0.0;
    if (!centroids.empty()) {
        mean_centroid = std::accumulate(centroids.begin(), centroids.end(), 0.0) / centroids.size();
        double var = 0.0;
        for (double v : centroids) var += (v - mean_centroid) * (v - mean_centroid);
        std_centroid = std::sqrt(var / centroids.size());
    }
    double zcr = compute_zero_crossing_rate(mono);
    double duration = static_cast<double>(mono.size()) / sample_rate_;
    double peak = 0.0;
    for (float s : mono) peak = std::max(peak, static_cast<double>(std::abs(s)));
    return {mean_rms, std_rms, mean_centroid, std_centroid, zcr, duration, peak};
}

double SimilarityDetector::cosine_similarity(
    const std::vector<double>& a, const std::vector<double>& b) const {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10 || norm_b < 1e-10) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

bool SimilarityDetector::are_similar(
    const std::vector<float>& audio_a,
    const std::vector<float>& audio_b,
    int channels) const {
    auto fa = compute_features(audio_a, channels);
    auto fb = compute_features(audio_b, channels);
    return cosine_similarity(fa, fb) >= threshold_;
}

std::vector<SimilarityResult> SimilarityDetector::find_duplicates(
    const std::vector<std::vector<float>>& clips,
    int channels) const {
    std::vector<std::vector<double>> features;
    features.reserve(clips.size());
    for (const auto& clip : clips) {
        features.push_back(compute_features(clip, channels));
    }
    std::vector<SimilarityResult> results;
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = i + 1; j < features.size(); ++j) {
            double score = cosine_similarity(features[i], features[j]);
            if (score >= threshold_) {
                results.push_back({
                    static_cast<int>(i),
                    static_cast<int>(j),
                    score,
                    true,
                });
            }
        }
    }
    return results;
}

} // namespace dsp
