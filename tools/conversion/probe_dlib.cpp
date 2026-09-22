#include <dlib/dnn.h>
#include <dlib/matrix.h>
#include <dlib/pixel.h>

#include "../network_definitions.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string task;
    std::string model;
    std::string images;
    std::string logits;
    std::string probabilities;
    std::string stages_dir;
    long count = 0;
};

long parse_positive(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < 1) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value(argv[++index]);
        if (argument == "--task") {
            options.task = value;
        } else if (argument == "--model") {
            options.model = value;
        } else if (argument == "--images") {
            options.images = value;
        } else if (argument == "--count") {
            options.count = parse_positive(value, "count");
        } else if (argument == "--logits") {
            options.logits = value;
        } else if (argument == "--probabilities") {
            options.probabilities = value;
        } else if (argument == "--stages-dir") {
            options.stages_dir = value;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if ((options.task != "age" && options.task != "gender") ||
        options.model.empty() || options.images.empty() || options.count == 0 ||
        options.logits.empty() || options.probabilities.empty()) {
        throw std::runtime_error(
            "required: --task age|gender --model FILE --images RGB --count N "
            "--logits FILE --probabilities FILE");
    }
    return options;
}

std::vector<dlib::matrix<dlib::rgb_pixel>> load_images(
    const std::string& path, long count, long size) {
    const std::uint64_t expected = static_cast<std::uint64_t>(count) * size * size * 3;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<std::uint64_t>(input.tellg()) != expected) {
        throw std::runtime_error("RGB input size does not match count*height*width*3");
    }
    input.seekg(0);
    std::vector<dlib::matrix<dlib::rgb_pixel>> images;
    images.reserve(static_cast<std::size_t>(count));
    for (long sample = 0; sample < count; ++sample) {
        dlib::matrix<dlib::rgb_pixel> image(size, size);
        for (long row = 0; row < size; ++row) {
            for (long column = 0; column < size; ++column) {
                unsigned char channels[3];
                input.read(reinterpret_cast<char*>(channels), 3);
                image(row, column) = dlib::rgb_pixel(channels[0], channels[1], channels[2]);
            }
        }
        images.push_back(std::move(image));
    }
    return images;
}

void write_tensor(const std::string& path, const dlib::tensor& value) {
    static const std::uint16_t endian_test = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_test) != 1) {
        throw std::runtime_error("probe output requires a little-endian host");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create output: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(value.host()),
        static_cast<std::streamsize>(value.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed while writing output: " + path);
    }
}

std::string join_path(const std::string& directory, const std::string& name) {
    if (directory.empty() || directory.back() == '/') {
        return directory + name;
    }
    return directory + "/" + name;
}

// dlib disables get_output() on any layer that an in-place layer is stacked on
// top of, so only indices whose immediate successor is not in-place can be read.
// Each entry is therefore the top of one in-place chain, which is also the
// tensor the converted graph names `layer_<index>_...`.
struct StageIndex {
    long index;
    const dlib::tensor* value;
};

void write_stage_manifest(
    const std::string& directory, const std::vector<StageIndex>& stages) {
    std::ofstream output(join_path(directory, "stages.json"), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create stage manifest in " + directory);
    }
    output << "[\n";
    for (std::size_t position = 0; position < stages.size(); ++position) {
        const dlib::tensor& value = *stages[position].value;
        output << "  {\"layer\": " << stages[position].index
               << ", \"file\": \"stage-" << stages[position].index << ".f32\""
               << ", \"shape\": [" << value.num_samples() << ", " << value.k()
               << ", " << value.nr() << ", " << value.nc() << "]}"
               << (position + 1 < stages.size() ? "," : "") << "\n";
    }
    output << "]\n";
    if (!output) {
        throw std::runtime_error("failed while writing the stage manifest");
    }
}

void write_stage_set(const std::string& directory, const std::vector<StageIndex>& stages) {
    for (const StageIndex& stage : stages) {
        write_tensor(
            join_path(directory, "stage-" + std::to_string(stage.index) + ".f32"),
            *stage.value);
    }
    write_stage_manifest(directory, stages);
}

void write_age_stages(
    const std::string& directory, age_gender_models::age_network& network) {
    const std::vector<StageIndex> stages = {
        {50, &dlib::layer<50>(network).get_output()},  // stem con+affine+relu
        {49, &dlib::layer<49>(network).get_output()},  // stem max pool
        {41, &dlib::layer<41>(network).get_output()},  // 64 residual output
        {32, &dlib::layer<32>(network).get_output()},  // 128 down skip branch
        {30, &dlib::layer<30>(network).get_output()},  // 128 down residual output
        {22, &dlib::layer<22>(network).get_output()},  // 128 residual output
        {13, &dlib::layer<13>(network).get_output()},  // 256 down skip branch
        {11, &dlib::layer<11>(network).get_output()},  // 256 down residual output
        {3, &dlib::layer<3>(network).get_output()},    // 256 residual output
        {2, &dlib::layer<2>(network).get_output()},    // global average pool
        {1, &dlib::layer<1>(network).get_output()},    // logits
    };
    write_stage_set(directory, stages);
}

void write_gender_stages(
    const std::string& directory, age_gender_models::gender_network& network) {
    const std::vector<StageIndex> stages = {
        {17, &dlib::layer<17>(network).get_output()},  // first 32-filter block
        {14, &dlib::layer<14>(network).get_output()},  // second 32-filter block
        {13, &dlib::layer<13>(network).get_output()},  // first average pool
        {10, &dlib::layer<10>(network).get_output()},  // first 64-filter block
        {7, &dlib::layer<7>(network).get_output()},    // second 64-filter block
        {2, &dlib::layer<2>(network).get_output()},    // multiply after fc(16)
        {1, &dlib::layer<1>(network).get_output()},    // logits
    };
    write_stage_set(directory, stages);
}

template <typename Network>
void write_stages(const std::string&, Network&) {}

template <>
void write_stages(
    const std::string& directory, age_gender_models::age_network& network) {
    write_age_stages(directory, network);
}

template <>
void write_stages(
    const std::string& directory, age_gender_models::gender_network& network) {
    write_gender_stages(directory, network);
}

template <typename Network>
void run_probe(const Options& options, long size) {
    Network network;
    dlib::deserialize(options.model) >> network;
    dlib::softmax<typename Network::subnet_type> softmax;
    softmax.subnet() = network.subnet();
    const auto images = load_images(options.images, options.count, size);
    const dlib::tensor& logits = network.subnet()(images.begin(), images.end());
    write_tensor(options.logits, logits);
    if (!options.stages_dir.empty()) {
        write_stages(options.stages_dir, network);
    }
    const dlib::tensor& probabilities = softmax(images.begin(), images.end());
    write_tensor(options.probabilities, probabilities);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.task == "age") {
            run_probe<age_gender_models::age_network>(options, 64);
        } else {
            run_probe<age_gender_models::gender_network>(options, 32);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << '\n';
        return 1;
    }
}
