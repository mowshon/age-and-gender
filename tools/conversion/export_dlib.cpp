#include <dlib/dnn.h>

#include "../network_definitions.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string age_model;
    std::string gender_model;
    std::string age_xml;
    std::string gender_xml;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--version") {
            std::cout << "age-and-gender dlib exporter 1\n"
                      << "dlib 19.20.0\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value(argv[++index]);
        if (argument == "--age-model") {
            options.age_model = value;
        } else if (argument == "--gender-model") {
            options.gender_model = value;
        } else if (argument == "--age-xml") {
            options.age_xml = value;
        } else if (argument == "--gender-xml") {
            options.gender_xml = value;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.age_model.empty() || options.gender_model.empty() ||
        options.age_xml.empty() || options.gender_xml.empty()) {
        throw std::runtime_error(
            "required: --age-model FILE --gender-model FILE "
            "--age-xml FILE --gender-xml FILE");
    }
    return options;
}

template <typename Network>
void export_network(const std::string& source, const std::string& destination) {
    Network network;
    dlib::deserialize(source) >> network;

    std::ofstream output(destination, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create XML output: " + destination);
    }
    output.imbue(std::locale::classic());
    dlib::net_to_xml(network, output);
    if (!output) {
        throw std::runtime_error("failed while writing XML output: " + destination);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::locale::global(std::locale::classic());
        const Options options = parse_options(argc, argv);
        export_network<age_gender_models::age_network>(options.age_model, options.age_xml);
        export_network<age_gender_models::gender_network>(
            options.gender_model, options.gender_xml);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "export error: " << error.what() << '\n';
        return 1;
    }
}
