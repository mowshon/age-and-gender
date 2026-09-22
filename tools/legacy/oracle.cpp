#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#ifndef ORACLE_BUILD_TYPE
#define ORACLE_BUILD_TYPE "unknown"
#endif

#ifndef ORACLE_CXX_FLAGS
#define ORACLE_CXX_FLAGS "unknown"
#endif

using namespace dlib;

namespace {

const unsigned long number_of_age_classes = 81;

// Keep these aliases structurally identical to src/main.cpp. They are the oracle.
template <int num_filters, template <typename> class BN, int stride, typename SUBNET>
using basicblock = BN<con<num_filters, 3, 3, 1, 1,
    relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

template <
    template <int, template <typename> class, int, typename> class BLOCK,
    int num_filters,
    template <typename> class BN,
    typename SUBNET>
using residual = add_prev1<BLOCK<num_filters, BN, 1, tag1<SUBNET>>>;

template <
    template <int, template <typename> class, int, typename> class BLOCK,
    int num_filters,
    template <typename> class BN,
    typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2,
    skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

template <
    template <template <int, template <typename> class, int, typename> class,
              int, template <typename> class, typename> class RESIDUAL,
    template <int, template <typename> class, int, typename> class BLOCK,
    int num_filters,
    template <typename> class BN,
    typename SUBNET>
using residual_block = relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

template <int num_filters, typename SUBNET>
using aresbasicblock_down =
    residual_block<residual_down, basicblock, num_filters, affine, SUBNET>;
template <typename SUBNET>
using aresbasicblock256 =
    residual_block<residual, basicblock, 256, affine, SUBNET>;
template <typename SUBNET>
using aresbasicblock128 =
    residual_block<residual, basicblock, 128, affine, SUBNET>;
template <typename SUBNET>
using aresbasicblock64 =
    residual_block<residual, basicblock, 64, affine, SUBNET>;

template <typename INPUT>
using aresnet_input = max_pool<3, 3, 2, 2,
    relu<affine<con<64, 7, 7, 2, 2, INPUT>>>>;
template <typename SUBNET>
using aresnet10_level1 = aresbasicblock256<aresbasicblock_down<256, SUBNET>>;
template <typename SUBNET>
using aresnet10_level2 = aresbasicblock128<aresbasicblock_down<128, SUBNET>>;
template <typename SUBNET>
using aresnet10_level3 = aresbasicblock64<SUBNET>;
template <typename INPUT>
using aresnet10_backbone = avg_pool_everything<aresnet10_level1<
    aresnet10_level2<aresnet10_level3<aresnet_input<INPUT>>>>>;
using apredictor_t = loss_multiclass_log<
    fc<number_of_age_classes, aresnet10_backbone<input_rgb_image>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, stride, stride,
    relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET>
using ares_ = relu<block<N, affine, 1, SUBNET>>;
template <typename SUBNET>
using alevel1 = avg_pool<2, 2, 2, 2, ares_<64, SUBNET>>;
template <typename SUBNET>
using alevel2 = avg_pool<2, 2, 2, 2, ares_<32, SUBNET>>;
using agender_type = loss_multiclass_log<fc<2, multiply<relu<fc<16,
    multiply<alevel1<alevel2<input_rgb_image_sized<32>>>>>>>>>;

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

long peak_rss_kib() {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
        return usage.ru_maxrss / 1024;
#else
        return usage.ru_maxrss;
#endif
    }
#endif
    return -1;
}

struct Options {
    std::string image_path;
    std::string model_dir;
    std::string output_dir;
    long width = 0;
    long height = 0;
    unsigned long benchmark_runs = 3;
    std::vector<rectangle> boxes;
};

struct PublicResult {
    std::string gender;
    int gender_confidence;
    int age;
    int age_confidence;
    rectangle face;
};

struct TimedRun {
    std::vector<PublicResult> results;
    double total_ms = 0;
    double input_copy_ms = 0;
    double subnet_copy_ms = 0;
    double frontend_ms = 0;
    double cnn_ms = 0;
};

struct StageFace {
    rectangle face;
    std::vector<point> landmarks;
    std::vector<float> gender_logits;
    std::vector<float> gender_probabilities;
    std::vector<float> age_logits;
    std::vector<float> age_probabilities;
    float age_expectation;
    PublicResult result;
};

std::string join_path(const std::string& directory, const std::string& name) {
    if (directory.empty() || directory.back() == '/') {
        return directory + name;
    }
    return directory + "/" + name;
}

long parse_long(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

rectangle parse_box(const std::string& value) {
    std::stringstream stream(value);
    std::string item;
    std::vector<long> coordinates;
    while (std::getline(stream, item, ',')) {
        coordinates.push_back(parse_long(item, "box coordinate"));
    }
    if (coordinates.size() != 4) {
        throw std::runtime_error("--box must be top,right,bottom,left");
    }
    return rectangle(coordinates[3], coordinates[0], coordinates[1], coordinates[2]);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--version") {
            std::cout << "age-and-gender legacy oracle 1\n"
                      << "dlib 19.20.0\n"
                      << "compiler " << __VERSION__ << "\n"
                      << "build_type " << ORACLE_BUILD_TYPE << "\n"
                      << "flags " << ORACLE_CXX_FLAGS << "\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value(argv[++index]);
        if (argument == "--image") {
            options.image_path = value;
        } else if (argument == "--width") {
            options.width = parse_long(value, "width");
        } else if (argument == "--height") {
            options.height = parse_long(value, "height");
        } else if (argument == "--models") {
            options.model_dir = value;
        } else if (argument == "--output") {
            options.output_dir = value;
        } else if (argument == "--benchmark-runs") {
            const long runs = parse_long(value, "benchmark runs");
            if (runs < 1) {
                throw std::runtime_error("--benchmark-runs must be positive");
            }
            options.benchmark_runs = static_cast<unsigned long>(runs);
        } else if (argument == "--box") {
            options.boxes.push_back(parse_box(value));
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.image_path.empty() || options.model_dir.empty() ||
        options.output_dir.empty() || options.width <= 0 || options.height <= 0) {
        throw std::runtime_error(
            "required: --image RAW_RGB --width W --height H --models DIR --output DIR");
    }
    return options;
}

matrix<rgb_pixel> load_raw_rgb(const Options& options) {
    const std::uint64_t expected = static_cast<std::uint64_t>(options.width) *
        static_cast<std::uint64_t>(options.height) * 3;
    std::ifstream input(options.image_path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open RGB input: " + options.image_path);
    }
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) != expected) {
        throw std::runtime_error("RGB input size does not match width*height*3");
    }
    input.seekg(0);
    matrix<rgb_pixel> image(options.height, options.width);
    for (long row = 0; row < options.height; ++row) {
        for (long column = 0; column < options.width; ++column) {
            unsigned char channels[3];
            input.read(reinterpret_cast<char*>(channels), 3);
            image(row, column) = rgb_pixel(channels[0], channels[1], channels[2]);
        }
    }
    return image;
}

std::vector<float> tensor_values(const tensor& value) {
    return std::vector<float>(value.host(), value.host() + value.size());
}

void write_bytes(const std::string& path, const void* data, std::size_t size) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create artifact: " + path);
    }
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("cannot write artifact: " + path);
    }
}

void write_chip(const std::string& path, const matrix<rgb_pixel>& chip) {
    std::vector<unsigned char> bytes;
    bytes.reserve(static_cast<std::size_t>(chip.size()) * 3);
    for (long row = 0; row < chip.nr(); ++row) {
        for (long column = 0; column < chip.nc(); ++column) {
            const rgb_pixel pixel = chip(row, column);
            bytes.push_back(pixel.red);
            bytes.push_back(pixel.green);
            bytes.push_back(pixel.blue);
        }
    }
    write_bytes(path, bytes.data(), bytes.size());
}

void write_tensor(const std::string& path, const tensor& value) {
    static const std::uint16_t endian_test = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_test) != 1) {
        throw std::runtime_error("oracle artifacts require a little-endian reference host");
    }
    write_bytes(path, value.host(), value.size() * sizeof(float));
}

float estimated_age(const std::vector<float>& probabilities, float& confidence) {
    float estimate = 0.25f * probabilities.at(0);
    confidence = probabilities.at(0);
    for (std::uint16_t index = 1; index < number_of_age_classes; ++index) {
        estimate += static_cast<float>(index) * probabilities.at(index);
        if (probabilities.at(index) > confidence) {
            confidence = probabilities.at(index);
        }
    }
    return estimate;
}

PublicResult make_result(
    const rectangle& face,
    const std::vector<float>& gender_probabilities,
    const std::vector<float>& age_probabilities,
    float& age_expectation) {
    const bool male = gender_probabilities.at(0) < gender_probabilities.at(1);
    const float gender_confidence = male
        ? gender_probabilities.at(1)
        : gender_probabilities.at(0);
    float age_confidence = 0;
    age_expectation = estimated_age(age_probabilities, age_confidence);
    return PublicResult{
        male ? "male" : "female",
        static_cast<int>(std::floor(gender_confidence * 100.0f)),
        static_cast<int>(std::lround(age_expectation)),
        static_cast<int>(std::floor(age_confidence * 100.0f)),
        face,
    };
}

std::vector<rectangle> select_faces(
    const matrix<rgb_pixel>& image,
    const std::vector<rectangle>& boxes,
    frontal_face_detector& detector) {
    return boxes.empty() ? detector(image) : boxes;
}

TimedRun legacy_predict(
    const matrix<rgb_pixel>& image,
    const std::vector<rectangle>& boxes,
    frontal_face_detector& detector,
    shape_predictor& predictor,
    agender_type& gender_net,
    apredictor_t& age_net) {
    TimedRun run;
    const auto total_start = Clock::now();

    // Match src/main.cpp::from_numpy(), which copies every RGB pixel per call.
    const auto input_copy_start = Clock::now();
    matrix<rgb_pixel> in(image.nr(), image.nc());
    for (long row = 0; row < image.nr(); ++row) {
        for (long column = 0; column < image.nc(); ++column) {
            const rgb_pixel pixel = image(row, column);
            in(row, column) = rgb_pixel(pixel.red, pixel.green, pixel.blue);
        }
    }
    run.input_copy_ms = milliseconds(input_copy_start, Clock::now());

    const auto copy_start = Clock::now();
    softmax<agender_type::subnet_type> gender_softmax;
    gender_softmax.subnet() = gender_net.subnet();
    softmax<apredictor_t::subnet_type> age_softmax;
    age_softmax.subnet() = age_net.subnet();
    const auto copy_end = Clock::now();
    run.subnet_copy_ms = milliseconds(copy_start, copy_end);

    auto frontend_start = Clock::now();
    const std::vector<rectangle> faces = select_faces(in, boxes, detector);
    run.frontend_ms += milliseconds(frontend_start, Clock::now());

    for (const rectangle& face : faces) {
        frontend_start = Clock::now();
        const full_object_detection shape = predictor(in, face);
        if (shape.num_parts() == 0) {
            run.frontend_ms += milliseconds(frontend_start, Clock::now());
            continue;
        }
        matrix<rgb_pixel> chip;
        extract_image_chip(in, get_face_chip_details(shape, 32), chip);
        run.frontend_ms += milliseconds(frontend_start, Clock::now());

        const auto gender_start = Clock::now();
        const matrix<float, 1, 2> gender_probabilities = mat(gender_softmax(chip));
        run.cnn_ms += milliseconds(gender_start, Clock::now());

        frontend_start = Clock::now();
        extract_image_chip(in, get_face_chip_details(shape, 64), chip);
        run.frontend_ms += milliseconds(frontend_start, Clock::now());

        const auto age_start = Clock::now();
        const matrix<float, 1, number_of_age_classes> age_probabilities =
            mat(age_softmax(chip));
        run.cnn_ms += milliseconds(age_start, Clock::now());

        float expectation = 0;
        run.results.push_back(make_result(
            face,
            std::vector<float>(gender_probabilities.begin(), gender_probabilities.end()),
            std::vector<float>(age_probabilities.begin(), age_probabilities.end()),
            expectation));
    }
    run.total_ms = milliseconds(total_start, Clock::now());
    return run;
}

std::vector<StageFace> export_stages(
    const Options& options,
    const matrix<rgb_pixel>& image,
    frontal_face_detector& detector,
    shape_predictor& predictor,
    agender_type& gender_net,
    apredictor_t& age_net) {
    softmax<agender_type::subnet_type> gender_softmax;
    gender_softmax.subnet() = gender_net.subnet();
    softmax<apredictor_t::subnet_type> age_softmax;
    age_softmax.subnet() = age_net.subnet();

    const std::vector<rectangle> faces = select_faces(image, options.boxes, detector);
    std::vector<StageFace> stages;
    for (const rectangle& face : faces) {
        const full_object_detection shape = predictor(image, face);
        if (shape.num_parts() == 0) {
            continue;
        }
        StageFace stage;
        stage.face = face;
        for (unsigned long index = 0; index < shape.num_parts(); ++index) {
            stage.landmarks.push_back(shape.part(index));
        }

        matrix<rgb_pixel> gender_chip;
        extract_image_chip(image, get_face_chip_details(shape, 32), gender_chip);
        const std::string prefix = "face-" + std::to_string(stages.size());
        write_chip(join_path(options.output_dir, prefix + "-gender-chip.rgb"), gender_chip);
        resizable_tensor gender_input;
        input_layer(gender_net).to_tensor(&gender_chip, &gender_chip + 1, gender_input);
        write_tensor(join_path(options.output_dir, prefix + "-gender-input.f32"), gender_input);
        stage.gender_logits = tensor_values(gender_net.subnet()(gender_chip));
        stage.gender_probabilities = tensor_values(gender_softmax(gender_chip));

        matrix<rgb_pixel> age_chip;
        extract_image_chip(image, get_face_chip_details(shape, 64), age_chip);
        write_chip(join_path(options.output_dir, prefix + "-age-chip.rgb"), age_chip);
        resizable_tensor age_input;
        input_layer(age_net).to_tensor(&age_chip, &age_chip + 1, age_input);
        write_tensor(join_path(options.output_dir, prefix + "-age-input.f32"), age_input);
        stage.age_logits = tensor_values(age_net.subnet()(age_chip));
        stage.age_probabilities = tensor_values(age_softmax(age_chip));
        stage.result = make_result(
            face,
            stage.gender_probabilities,
            stage.age_probabilities,
            stage.age_expectation);
        stages.push_back(stage);
    }
    return stages;
}

void write_float_array(std::ostream& output, const std::vector<float>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec << std::setfill(' ');
                } else {
                    output << character;
                }
        }
    }
    output << '"';
    return output.str();
}

void write_public_result(std::ostream& output, const PublicResult& result) {
    output << "{\"gender\":{\"value\":\"" << result.gender
           << "\",\"confidence\":" << result.gender_confidence
           << "},\"age\":{\"value\":" << result.age
           << ",\"confidence\":" << result.age_confidence
           << "},\"face\":[" << result.face.left() << ',' << result.face.top()
           << ',' << result.face.right() << ',' << result.face.bottom() << "]}";
}

void write_report(
    const Options& options,
    double model_load_ms,
    double input_load_ms,
    const TimedRun& cold,
    const std::vector<TimedRun>& warm,
    long inference_peak_rss_kib,
    double export_ms,
    agender_type& gender_net,
    apredictor_t& age_net,
    const std::vector<StageFace>& stages) {
    std::ofstream output(join_path(options.output_dir, "oracle-report.json"));
    if (!output) {
        throw std::runtime_error("cannot create oracle-report.json");
    }
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"reference\": {\"dlib\": \"19.20.0\", \"device\": \"cpu\", "
              "\"float_mode\": \"float32-no-fast-math\", \"detector_upsampling\": 0},\n"
           << "  \"models\": {\"gender_input_means_rgb\":["
           << input_layer(gender_net).get_avg_red() << ','
           << input_layer(gender_net).get_avg_green() << ','
           << input_layer(gender_net).get_avg_blue() << "],\"age_input_means_rgb\":["
           << input_layer(age_net).get_avg_red() << ','
           << input_layer(age_net).get_avg_green() << ','
           << input_layer(age_net).get_avg_blue() << "]},\n"
           << "  \"input\": {\"path\": " << json_string(options.image_path)
           << ", \"width\": " << options.width << ", \"height\": "
           << options.height << ", \"layout\": \"HWC RGB uint8\"},\n"
           << "  \"box_mode\": \"" << (options.boxes.empty() ? "detect" : "explicit")
           << "\",\n  \"input_boxes_trbl\": [";
    for (std::size_t index = 0; index < options.boxes.size(); ++index) {
        if (index != 0) output << ',';
        const rectangle& box = options.boxes[index];
        output << '[' << box.top() << ',' << box.right() << ',' << box.bottom()
               << ',' << box.left() << ']';
    }
    output << "],\n"
           << "  \"timing_ms\": {\"model_load\": " << model_load_ms
           << ", \"input_load\": " << input_load_ms
           << ", \"cold_total\": " << cold.total_ms
           << ", \"cold_input_copy\": " << cold.input_copy_ms
           << ", \"cold_subnet_copy\": " << cold.subnet_copy_ms
           << ", \"cold_frontend\": " << cold.frontend_ms
           << ", \"cold_cnn\": " << cold.cnn_ms
           << ", \"warm_total\": [";
    for (std::size_t index = 0; index < warm.size(); ++index) {
        if (index != 0) output << ',';
        output << warm[index].total_ms;
    }
    output << "], \"warm_input_copy\": [";
    for (std::size_t index = 0; index < warm.size(); ++index) {
        if (index != 0) output << ',';
        output << warm[index].input_copy_ms;
    }
    output << "], \"warm_subnet_copy\": [";
    for (std::size_t index = 0; index < warm.size(); ++index) {
        if (index != 0) output << ',';
        output << warm[index].subnet_copy_ms;
    }
    output << "], \"warm_frontend\": [";
    for (std::size_t index = 0; index < warm.size(); ++index) {
        if (index != 0) output << ',';
        output << warm[index].frontend_ms;
    }
    output << "], \"warm_cnn\": [";
    for (std::size_t index = 0; index < warm.size(); ++index) {
        if (index != 0) output << ',';
        output << warm[index].cnn_ms;
    }
    output << "], \"instrumented_export\": " << export_ms << "},\n"
           << "  \"inference_peak_rss_kib\": " << inference_peak_rss_kib << ",\n"
           << "  \"process_peak_rss_kib\": " << peak_rss_kib() << ",\n";
    output << "  \"faces\": [\n";
    for (std::size_t index = 0; index < stages.size(); ++index) {
        if (index != 0) output << ",\n";
        const StageFace& stage = stages[index];
        output << "    {\"index\":" << index << ",\"rectangle\":["
               << stage.face.left() << ',' << stage.face.top() << ','
               << stage.face.right() << ',' << stage.face.bottom() << "],\"landmarks\":[";
        for (std::size_t point_index = 0; point_index < stage.landmarks.size(); ++point_index) {
            if (point_index != 0) output << ',';
            output << '[' << stage.landmarks[point_index].x() << ','
                   << stage.landmarks[point_index].y() << ']';
        }
        output << "],\"artifacts\":{\"gender_chip\":{\"path\":\"face-" << index
               << "-gender-chip.rgb\",\"dtype\":\"uint8\",\"shape\":[32,32,3]},"
                  "\"gender_input\":{\"path\":\"face-" << index
               << "-gender-input.f32\",\"dtype\":\"float32-le\",\"shape\":[1,3,32,32]},"
                  "\"age_chip\":{\"path\":\"face-" << index
               << "-age-chip.rgb\",\"dtype\":\"uint8\",\"shape\":[64,64,3]},"
                  "\"age_input\":{\"path\":\"face-" << index
               << "-age-input.f32\",\"dtype\":\"float32-le\",\"shape\":[1,3,64,64]}},"
                  "\"gender_logits\":";
        write_float_array(output, stage.gender_logits);
        output << ",\"gender_probabilities\":";
        write_float_array(output, stage.gender_probabilities);
        output << ",\"age_logits\":";
        write_float_array(output, stage.age_logits);
        output << ",\"age_probabilities\":";
        write_float_array(output, stage.age_probabilities);
        output << ",\"age_expectation\":" << stage.age_expectation
               << ",\"result\":";
        write_public_result(output, stage.result);
        output << '}';
    }
    output << "\n  ],\n  \"results\": [";
    for (std::size_t index = 0; index < stages.size(); ++index) {
        if (index != 0) output << ',';
        write_public_result(output, stages[index].result);
    }
    output << "]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto model_start = Clock::now();
        apredictor_t age_net;
        agender_type gender_net;
        shape_predictor predictor;
        deserialize(join_path(options.model_dir, "dnn_age_predictor_v1.dat")) >> age_net;
        deserialize(join_path(options.model_dir, "dnn_gender_classifier_v1.dat")) >> gender_net;
        deserialize(join_path(options.model_dir, "shape_predictor_5_face_landmarks.dat")) >> predictor;
        frontal_face_detector detector = get_frontal_face_detector();
        const double model_load_ms = milliseconds(model_start, Clock::now());

        const auto input_start = Clock::now();
        const matrix<rgb_pixel> image = load_raw_rgb(options);
        const double input_load_ms = milliseconds(input_start, Clock::now());

        const TimedRun cold = legacy_predict(
            image, options.boxes, detector, predictor, gender_net, age_net);
        std::vector<TimedRun> warm;
        for (unsigned long index = 0; index < options.benchmark_runs; ++index) {
            warm.push_back(legacy_predict(
                image, options.boxes, detector, predictor, gender_net, age_net));
        }
        const long inference_peak_rss_kib = peak_rss_kib();

        const auto export_start = Clock::now();
        const std::vector<StageFace> stages = export_stages(
            options, image, detector, predictor, gender_net, age_net);
        const double export_ms = milliseconds(export_start, Clock::now());
        write_report(
            options,
            model_load_ms,
            input_load_ms,
            cold,
            warm,
            inference_peak_rss_kib,
            export_ms,
            gender_net,
            age_net,
            stages);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "legacy oracle: " << error.what() << '\n';
        return 1;
    }
}
