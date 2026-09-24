// Runs the real dlib shape predictor and face-chip extraction on one explicit
// (image, rectangle) pair, dumping per-cascade intermediate state alongside
// the final landmarks and chips. This is the stage-trace oracle spec/PR-8.md's
// "Initial feasibility deliverable" item 2 asks for ("compares every cascade
// stage with the dlib oracle").
//
// The per-cascade loop below is a straight inlining of
// dlib::shape_predictor::operator()'s body (shape_predictor.h:338-363), calling
// dlib's own dlib::impl:: helper functions (extract_feature_pixel_values,
// unnormalizing_tform, location) directly rather than reimplementing them, so
// the trace is guaranteed bit-identical to what the real predictor computes,
// not a second, possibly-diverging reconstruction of the same formulas. Chip
// extraction uses the unmodified public dlib::get_face_chip_details and
// dlib::extract_image_chip.
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_transforms/interpolation.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string image;
    long width = 0;
    long height = 0;
    long left = 0, top = 0, right = 0, bottom = 0;
    bool has_rect = false;
    long gender_size = 32;
    long age_size = 64;
    double padding = 0.2;
    std::string output_dir;
};

long parse_long(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    return parsed;
}

void parse_rect(Options& options, const std::string& value) {
    std::istringstream stream(value);
    std::string token;
    std::vector<long> parts;
    while (std::getline(stream, token, ',')) {
        parts.push_back(parse_long(token, "--rect"));
    }
    if (parts.size() != 4) {
        throw std::runtime_error("--rect requires left,top,right,bottom");
    }
    options.left = parts[0];
    options.top = parts[1];
    options.right = parts[2];
    options.bottom = parts[3];
    options.has_rect = true;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--version") {
            std::cout << "age-and-gender shape predictor probe 1\n"
                      << "dlib 19.20.0\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value(argv[++index]);
        if (argument == "--model") {
            options.model = value;
        } else if (argument == "--image") {
            options.image = value;
        } else if (argument == "--width") {
            options.width = parse_long(value, "--width");
        } else if (argument == "--height") {
            options.height = parse_long(value, "--height");
        } else if (argument == "--rect") {
            parse_rect(options, value);
        } else if (argument == "--gender-size") {
            options.gender_size = parse_long(value, "--gender-size");
        } else if (argument == "--age-size") {
            options.age_size = parse_long(value, "--age-size");
        } else if (argument == "--padding") {
            options.padding = std::stod(value);
        } else if (argument == "--output-dir") {
            options.output_dir = value;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.model.empty() || options.image.empty() || options.width <= 0 ||
        options.height <= 0 || !options.has_rect || options.output_dir.empty()) {
        throw std::runtime_error(
            "required: --model FILE --image RGB --width W --height H "
            "--rect L,T,R,B --output-dir DIR");
    }
    return options;
}

std::string join_path(const std::string& directory, const std::string& name) {
    if (directory.empty() || directory.back() == '/') {
        return directory + name;
    }
    return directory + "/" + name;
}

void require_little_endian() {
    static const std::uint16_t endian_test = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_test) != 1) {
        throw std::runtime_error("probe requires a little-endian host");
    }
}

dlib::matrix<dlib::rgb_pixel> load_image(const std::string& path, long width, long height) {
    const std::uint64_t expected = static_cast<std::uint64_t>(width) * height * 3;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<std::uint64_t>(input.tellg()) != expected) {
        throw std::runtime_error("RGB input size does not match width*height*3");
    }
    input.seekg(0);
    dlib::matrix<dlib::rgb_pixel> image(height, width);
    for (long row = 0; row < height; ++row) {
        for (long column = 0; column < width; ++column) {
            unsigned char channels[3];
            input.read(reinterpret_cast<char*>(channels), 3);
            image(row, column) = dlib::rgb_pixel(channels[0], channels[1], channels[2]);
        }
    }
    return image;
}

// Same private-field replication as export_shape_predictor.cpp; kept as an
// independent copy here rather than shared, matching this repo's existing
// export_dlib.cpp/probe_dlib.cpp convention of self-contained tool sources.
struct RawShapePredictor {
    dlib::matrix<float, 0, 1> initial_shape;
    std::vector<std::vector<dlib::impl::regression_tree>> forests;
    std::vector<std::vector<unsigned long>> anchor_idx;
    std::vector<std::vector<dlib::vector<float, 2>>> deltas;
};

RawShapePredictor load_raw(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open model: " + path);
    }
    int version = 0;
    dlib::deserialize(version, input);
    if (version != 1) {
        throw std::runtime_error(
            "unexpected shape_predictor version: " + std::to_string(version));
    }
    RawShapePredictor raw;
    dlib::deserialize(raw.initial_shape, input);
    dlib::deserialize(raw.forests, input);
    dlib::deserialize(raw.anchor_idx, input);
    dlib::deserialize(raw.deltas, input);
    return raw;
}

void write_f32_vector(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create output: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed while writing output: " + path);
    }
}

void write_f32_matrix(const std::string& path, const dlib::matrix<float, 0, 1>& value) {
    std::vector<float> flat(value.size());
    for (long i = 0; i < value.size(); ++i) {
        flat[static_cast<std::size_t>(i)] = value(i);
    }
    write_f32_vector(path, flat);
}

void write_rgb_image(const std::string& path, const dlib::array2d<dlib::rgb_pixel>& image) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create output: " + path);
    }
    for (long row = 0; row < image.nr(); ++row) {
        for (long column = 0; column < image.nc(); ++column) {
            const dlib::rgb_pixel& pixel = image[row][column];
            const unsigned char channels[3] = {pixel.red, pixel.green, pixel.blue};
            output.write(reinterpret_cast<const char*>(channels), 3);
        }
    }
    if (!output) {
        throw std::runtime_error("failed while writing output: " + path);
    }
}

void write_chip_details_json(
    const std::string& path, const dlib::chip_details& details) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create output: " + path);
    }
    output.precision(17);
    output << "{\n"
           << "  \"rect\": [" << details.rect.left() << ", " << details.rect.top() << ", "
           << details.rect.right() << ", " << details.rect.bottom() << "],\n"
           << "  \"angle\": " << details.angle << ",\n"
           << "  \"rows\": " << details.rows << ",\n"
           << "  \"cols\": " << details.cols << "\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error("failed while writing output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require_little_endian();
        const Options options = parse_options(argc, argv);
        const dlib::matrix<dlib::rgb_pixel> image =
            load_image(options.image, options.width, options.height);
        const RawShapePredictor raw = load_raw(options.model);
        const dlib::rectangle rect(options.left, options.top, options.right, options.bottom);

        dlib::matrix<float, 0, 1> current_shape = raw.initial_shape;
        std::vector<float> feature_pixel_values;
        const std::size_t num_cascades = raw.forests.size();
        for (std::size_t iter = 0; iter < num_cascades; ++iter) {
            dlib::impl::extract_feature_pixel_values(
                image, rect, current_shape, raw.initial_shape, raw.anchor_idx[iter],
                raw.deltas[iter], feature_pixel_values);
            write_f32_vector(
                join_path(options.output_dir, "stage-" + std::to_string(iter) + "-features.f32"),
                feature_pixel_values);
            for (const dlib::impl::regression_tree& tree : raw.forests[iter]) {
                unsigned long leaf_idx = 0;
                current_shape += tree(feature_pixel_values, leaf_idx);
            }
            write_f32_matrix(
                join_path(options.output_dir, "stage-" + std::to_string(iter) + "-shape.f32"),
                current_shape);
        }

        const dlib::point_transform_affine tform_to_img = dlib::impl::unnormalizing_tform(rect);
        const long num_parts = current_shape.size() / 2;
        std::vector<dlib::point> parts(static_cast<std::size_t>(num_parts));
        for (long i = 0; i < num_parts; ++i) {
            parts[static_cast<std::size_t>(i)] =
                tform_to_img(dlib::impl::location(current_shape, static_cast<unsigned long>(i)));
        }
        const dlib::full_object_detection det(rect, parts);

        {
            std::ofstream output(join_path(options.output_dir, "landmarks.json"), std::ios::binary);
            output << "{\n  \"rectangle\": [" << rect.left() << ", " << rect.top() << ", "
                   << rect.right() << ", " << rect.bottom() << "],\n  \"landmarks\": [\n";
            for (long i = 0; i < num_parts; ++i) {
                output << "    [" << parts[static_cast<std::size_t>(i)].x() << ", "
                       << parts[static_cast<std::size_t>(i)].y() << "]"
                       << (i + 1 < num_parts ? ",\n" : "\n");
            }
            output << "  ],\n  \"num_cascades\": " << num_cascades << "\n}\n";
        }

        const dlib::chip_details gender_details =
            dlib::get_face_chip_details(det, static_cast<unsigned long>(options.gender_size), options.padding);
        const dlib::chip_details age_details =
            dlib::get_face_chip_details(det, static_cast<unsigned long>(options.age_size), options.padding);
        write_chip_details_json(join_path(options.output_dir, "chip-gender-details.json"), gender_details);
        write_chip_details_json(join_path(options.output_dir, "chip-age-details.json"), age_details);

        dlib::array2d<dlib::rgb_pixel> gender_chip, age_chip;
        dlib::extract_image_chip(image, gender_details, gender_chip);
        dlib::extract_image_chip(image, age_details, age_chip);
        write_rgb_image(join_path(options.output_dir, "chip-gender.rgb"), gender_chip);
        write_rgb_image(join_path(options.output_dir, "chip-age.rgb"), age_chip);

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << '\n';
        return 1;
    }
}
