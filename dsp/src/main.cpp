#include "dsp_engine.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <command> [options]\n"
              << "\nCommands:\n"
              << "  process       Extract and process a loop from raw PCM input\n"
              << "  crossfade     Crossfade two raw PCM clips together\n"
              << "  similarity    Compute similarity score between two clips\n"
              << "\nOptions:\n"
              << "  --sample-rate INT      Sample rate (default: 44100)\n"
              << "  --channels INT         Number of channels (default: 2)\n"
              << "  --bpm FLOAT            BPM for rhythmic processing (default: 120.0)\n"
              << "  --bars INT             Number of bars (default: 4)\n"
              << "  --crossfade-ms FLOAT   Crossfade duration in ms (default: 50.0)\n"
              << "  --similarity-thresh F  Similarity threshold (default: 0.85)\n"
              << "  --fade-shape STR       Fade shape: linear|equal_power|logarithmic\n"
              << "  --input FILE           Input raw PCM float32 file\n"
              << "  --input-a FILE         First input file (for crossfade/similarity)\n"
              << "  --input-b FILE         Second input file (for crossfade/similarity)\n"
              << "  --output FILE          Output raw PCM float32 file\n"
              << std::flush;
}

static std::vector<float> read_pcm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open input file: " + path);
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static void write_pcm(const std::string& path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open output file: " + path);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    dsp::EngineConfig cfg;
    std::string input_file, input_a, input_b, output_file;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing argument for " + arg);
            return argv[++i];
        };
        if (arg == "--sample-rate") cfg.sample_rate = std::stoi(next());
        else if (arg == "--channels") cfg.channels = std::stoi(next());
        else if (arg == "--bpm") cfg.bpm = std::stod(next());
        else if (arg == "--bars") cfg.bars = std::stoi(next());
        else if (arg == "--crossfade-ms") cfg.crossfade_ms = std::stod(next());
        else if (arg == "--similarity-thresh") cfg.similarity_threshold = std::stod(next());
        else if (arg == "--fade-shape") cfg.fade_shape = next();
        else if (arg == "--input") input_file = next();
        else if (arg == "--input-a") input_a = next();
        else if (arg == "--input-b") input_b = next();
        else if (arg == "--output") output_file = next();
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        dsp::DspEngine engine(cfg);

        if (command == "process") {
            if (input_file.empty()) throw std::runtime_error("--input required for process command");
            auto audio = read_pcm(input_file);
            auto result = engine.process(audio);
            if (!result.success) throw std::runtime_error(result.error_message);
            if (!output_file.empty()) write_pcm(output_file, result.audio);
            std::cout << "beats=" << result.beat_positions.size()
                      << " samples=" << result.audio.size()
                      << " success=true\n";

        } else if (command == "crossfade") {
            if (input_a.empty() || input_b.empty())
                throw std::runtime_error("--input-a and --input-b required for crossfade command");
            auto audio_a = read_pcm(input_a);
            auto audio_b = read_pcm(input_b);
            auto result = engine.process_and_crossfade(audio_a, audio_b);
            if (!result.success) throw std::runtime_error(result.error_message);
            if (!output_file.empty()) write_pcm(output_file, result.audio);
            std::cout << "samples=" << result.audio.size()
                      << " similarity=" << result.similarity_score
                      << " duplicate=" << (result.is_duplicate ? "true" : "false")
                      << "\n";

        } else if (command == "similarity") {
            if (input_a.empty() || input_b.empty())
                throw std::runtime_error("--input-a and --input-b required for similarity command");
            auto audio_a = read_pcm(input_a);
            auto audio_b = read_pcm(input_b);
            auto sim = engine.check_similarity(audio_a, audio_b);
            std::cout << "score=" << sim.score
                      << " duplicate=" << (sim.is_duplicate ? "true" : "false")
                      << "\n";

        } else {
            std::cerr << "Unknown command: " << command << "\n";
            print_usage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
