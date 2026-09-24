// Dumps dlib::shape_predictor's trained cascade parameters to a flat,
// portable format so a Python/NumPy port can load them without linking dlib.
//
// dlib::shape_predictor keeps initial_shape/forests/anchor_idx/deltas private,
// reachable only through its own `friend serialize/deserialize` free functions
// (dlib/image_processing/shape_predictor.h). Rather than modifying vendored
// dlib to add accessors, this file replicates that exact deserialize sequence
// (same field order, same generic dlib::deserialize overloads for matrix/
// vector/std::vector) into local variables, then writes them out itself. See
// spec/PR-8.md's "Initial feasibility deliverable" and shape_predictor.h:287-431.
#include <dlib/image_processing/shape_predictor.h>

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
    std::string output_dir;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--version") {
            std::cout << "age-and-gender shape predictor exporter 1\n"
                      << "dlib 19.20.0\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value(argv[++index]);
        if (argument == "--model") {
            options.model = value;
        } else if (argument == "--output-dir") {
            options.output_dir = value;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.model.empty() || options.output_dir.empty()) {
        throw std::runtime_error("required: --model FILE --output-dir DIR");
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
        throw std::runtime_error("export requires a little-endian host");
    }
}

class BinaryWriter {
public:
    explicit BinaryWriter(const std::string& path) : stream_(path, std::ios::binary) {
        if (!stream_) {
            throw std::runtime_error("cannot create output: " + path);
        }
    }

    void write_u32(std::uint32_t value) {
        stream_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    void write_f32(float value) {
        stream_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    void finish(const std::string& path) {
        if (!stream_) {
            throw std::runtime_error("failed while writing output: " + path);
        }
    }

private:
    std::ofstream stream_;
};

// Mirrors dlib::deserialize(shape_predictor&, istream&) field-for-field
// (shape_predictor.h:421-431), but into plain local containers instead of a
// shape_predictor's private members, since those are only reachable from
// dlib's own friend functions.
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

void write_initial_shape(
    const std::string& output_dir, const dlib::matrix<float, 0, 1>& initial_shape) {
    BinaryWriter writer(join_path(output_dir, "initial_shape.f32"));
    for (long index = 0; index < initial_shape.size(); ++index) {
        writer.write_f32(initial_shape(index));
    }
    writer.finish("initial_shape.f32");
}

void write_anchor_idx_and_deltas(
    const std::string& output_dir,
    const std::vector<std::vector<unsigned long>>& anchor_idx,
    const std::vector<std::vector<dlib::vector<float, 2>>>& deltas) {
    BinaryWriter anchor_writer(join_path(output_dir, "anchor_idx.u32"));
    BinaryWriter delta_writer(join_path(output_dir, "deltas.f32"));
    for (std::size_t cascade = 0; cascade < anchor_idx.size(); ++cascade) {
        for (unsigned long idx : anchor_idx[cascade]) {
            anchor_writer.write_u32(static_cast<std::uint32_t>(idx));
        }
        for (const dlib::vector<float, 2>& delta : deltas[cascade]) {
            delta_writer.write_f32(delta.x());
            delta_writer.write_f32(delta.y());
        }
    }
    anchor_writer.finish("anchor_idx.u32");
    delta_writer.finish("deltas.f32");
}

// Returns, per cascade, the split count of each of its trees (leaves ==
// splits + 1, so this fully determines tree shape alongside num_parts).
std::vector<std::vector<std::uint32_t>> write_forests(
    const std::string& output_dir,
    const std::vector<std::vector<dlib::impl::regression_tree>>& forests) {
    BinaryWriter idx1_writer(join_path(output_dir, "splits_idx1.u32"));
    BinaryWriter idx2_writer(join_path(output_dir, "splits_idx2.u32"));
    BinaryWriter thresh_writer(join_path(output_dir, "splits_thresh.f32"));
    BinaryWriter leaves_writer(join_path(output_dir, "leaves.f32"));

    std::vector<std::vector<std::uint32_t>> tree_split_counts(forests.size());
    for (std::size_t cascade = 0; cascade < forests.size(); ++cascade) {
        tree_split_counts[cascade].reserve(forests[cascade].size());
        for (const dlib::impl::regression_tree& tree : forests[cascade]) {
            tree_split_counts[cascade].push_back(
                static_cast<std::uint32_t>(tree.splits.size()));
            for (const dlib::impl::split_feature& split : tree.splits) {
                idx1_writer.write_u32(static_cast<std::uint32_t>(split.idx1));
                idx2_writer.write_u32(static_cast<std::uint32_t>(split.idx2));
                thresh_writer.write_f32(split.thresh);
            }
            for (const dlib::matrix<float, 0, 1>& leaf : tree.leaf_values) {
                for (long index = 0; index < leaf.size(); ++index) {
                    leaves_writer.write_f32(leaf(index));
                }
            }
        }
    }
    idx1_writer.finish("splits_idx1.u32");
    idx2_writer.finish("splits_idx2.u32");
    thresh_writer.finish("splits_thresh.f32");
    leaves_writer.finish("leaves.f32");
    return tree_split_counts;
}

void write_manifest(
    const std::string& output_dir,
    const RawShapePredictor& raw,
    const std::vector<std::vector<std::uint32_t>>& tree_split_counts) {
    std::ofstream output(join_path(output_dir, "manifest.json"), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create manifest.json in " + output_dir);
    }
    const long num_parts = raw.initial_shape.size() / 2;
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"num_parts\": " << num_parts << ",\n";
    output << "  \"num_cascades\": " << raw.forests.size() << ",\n";
    output << "  \"cascade_num_features\": [";
    for (std::size_t cascade = 0; cascade < raw.anchor_idx.size(); ++cascade) {
        output << raw.anchor_idx[cascade].size()
               << (cascade + 1 < raw.anchor_idx.size() ? ", " : "");
    }
    output << "],\n";
    output << "  \"cascade_tree_splits\": [\n";
    for (std::size_t cascade = 0; cascade < tree_split_counts.size(); ++cascade) {
        output << "    [";
        for (std::size_t tree = 0; tree < tree_split_counts[cascade].size(); ++tree) {
            output << tree_split_counts[cascade][tree]
                   << (tree + 1 < tree_split_counts[cascade].size() ? ", " : "");
        }
        output << "]" << (cascade + 1 < tree_split_counts.size() ? ",\n" : "\n");
    }
    output << "  ]\n";
    output << "}\n";
    if (!output) {
        throw std::runtime_error("failed while writing manifest.json");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require_little_endian();
        const Options options = parse_options(argc, argv);
        const RawShapePredictor raw = load_raw(options.model);
        write_initial_shape(options.output_dir, raw.initial_shape);
        write_anchor_idx_and_deltas(options.output_dir, raw.anchor_idx, raw.deltas);
        const std::vector<std::vector<std::uint32_t>> tree_split_counts =
            write_forests(options.output_dir, raw.forests);
        write_manifest(options.output_dir, raw, tree_split_counts);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "export error: " << error.what() << '\n';
        return 1;
    }
}
