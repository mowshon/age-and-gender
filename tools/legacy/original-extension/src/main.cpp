#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>

#include "dlib/data_io.h"
#include "dlib/string.h"
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

namespace py = pybind11;
using namespace dlib;

const char* VERSION = "1.0.1";

// Age Predictor
// ----------------------------------------------------------------------------------------

// This block of statements defines a Resnet-10 architecture for the age predictor.
// We will use 81 classes (0-80 years old) to predict the age of a face.
const unsigned long number_of_age_classes = 81;

// The resnet basic block.
template<
	int num_filters,
	template<typename> class BN,  // some kind of batch normalization or affine layer
	int stride,
	typename SUBNET
>
using basicblock = BN<con<num_filters, 3, 3, 1, 1, relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

// A residual making use of the skip layer mechanism.
template<
	template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
	int num_filters,
	template<typename> class BN,  // some kind of batch normalization or affine layer
	typename SUBNET
> // adds the block to the result of tag1 (the subnet)
using residual = add_prev1<BLOCK<num_filters, BN, 1, tag1<SUBNET>>>;

// A residual that does subsampling (we need to subsample the output of the subnet, too).
template<
	template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
	int num_filters,
	template<typename> class BN,
	typename SUBNET
>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

// Residual block with optional downsampling and batch normalization.
template<
	template<template<int, template<typename> class, int, typename> class, int, template<typename>class, typename> class RESIDUAL,
	template<int, template<typename> class, int, typename> class BLOCK,
	int num_filters,
	template<typename> class BN,
	typename SUBNET
>
using residual_block = relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

template<int num_filters, typename SUBNET>
using aresbasicblock_down = residual_block<residual_down, basicblock, num_filters, affine, SUBNET>;

// Some useful definitions to design the affine versions for inference.
template<typename SUBNET> using aresbasicblock256 = residual_block<residual, basicblock, 256, affine, SUBNET>;
template<typename SUBNET> using aresbasicblock128 = residual_block<residual, basicblock, 128, affine, SUBNET>;
template<typename SUBNET> using aresbasicblock64  = residual_block<residual, basicblock, 64, affine, SUBNET>;

// Common input for standard resnets.
template<typename INPUT>
using aresnet_input = max_pool<3, 3, 2, 2, relu<affine<con<64, 7, 7, 2, 2, INPUT>>>>;

// Resnet-10 architecture for estimating.
template<typename SUBNET>
using aresnet10_level1 = aresbasicblock256<aresbasicblock_down<256, SUBNET>>;
template<typename SUBNET>
using aresnet10_level2 = aresbasicblock128<aresbasicblock_down<128, SUBNET>>;
template<typename SUBNET>
using aresnet10_level3 = aresbasicblock64<SUBNET>;
// The resnet 10 backbone.
template<typename INPUT>
using aresnet10_backbone = avg_pool_everything<
	aresnet10_level1<
	aresnet10_level2<
	aresnet10_level3<
	aresnet_input<INPUT>>>>>;

using apredictor_t = loss_multiclass_log<fc<number_of_age_classes, aresnet10_backbone<input_rgb_image>>>;

// Helper function to estimage the age
uint8_t get_estimated_age(matrix<float, 1, number_of_age_classes>& p, float& confidence)
{
	float estimated_age = (0.25f * p(0));
	confidence = p(0);

	for (uint16_t i = 1; i < number_of_age_classes; i++) {
		estimated_age += (static_cast<float>(i) * p(i));
		if (p(i) > confidence) confidence = p(i);
	}

	return std::lround(estimated_age);
}

// ----------------------------------

// This block of statements defines the network as proposed for the CNN Model I.
// We however removed the "dropout" regularization on the activations of convolutional layers.
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, stride, stride, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using res_ = relu<block<N, bn_con, 1, SUBNET>>;
template <int N, typename SUBNET> using ares_ = relu<block<N, affine, 1, SUBNET>>;

template <typename SUBNET> using alevel1 = avg_pool<2, 2, 2, 2, ares_<64, SUBNET>>;
template <typename SUBNET> using alevel2 = avg_pool<2, 2, 2, 2, ares_<32, SUBNET>>;

using agender_type = loss_multiclass_log<fc<2, multiply<relu<fc<16, multiply<alevel1<alevel2< input_rgb_image_sized<32>>>>>>>>>;

enum label_ : uint16_t
{
	female_label,
	male_label,
};

// ----------------------------------

// convert a numpy array into a dlib::matrix<dlib::rgb_pixel>
const dlib::matrix<dlib::rgb_pixel> from_numpy(const py::array_t<unsigned char>& ndarray) {
    const int num_channels = 3;
    auto raw = ndarray.unchecked<num_channels>();
    std::vector<unsigned char> rgb(3);
    dlib::matrix<dlib::rgb_pixel> image(raw.shape(0), raw.shape(1));
    for (ssize_t i = 0; i < raw.shape(0); i++)
    {
        for (ssize_t j = 0; j < raw.shape(1); j++)
        {
            image(i, j) = dlib::rgb_pixel(
                static_cast<unsigned char>(raw(i, j, 0)),
                static_cast<unsigned char>(raw(i, j, 1)),
                static_cast<unsigned char>(raw(i, j, 2)));
        }
    }
    return image;
}

class AgeAndGender {
	
	public:
		virtual ~AgeAndGender() { }
		virtual void load_shape_predictor(std::string filename);
		virtual void load_dnn_gender_classifier(std::string filename);
		virtual void load_dnn_age_predictor(std::string filename);
		virtual py::list predict(
			const py::array_t<unsigned char>& photo_numpy_array,
			py::list face_bounding_boxes
		);
		virtual std::vector<dlib::rectangle> from_py_list_with_tuple_to_vector_with_rectangles(py::list face_bounding_boxes);
	
	private:
      shape_predictor sp;
      agender_type gender_predictor_net;
      apredictor_t age_predictor_net;
      frontal_face_detector detector;
};

void AgeAndGender::load_shape_predictor(std::string filename) {
	detector = get_frontal_face_detector();
	deserialize(filename) >> sp;
}

void AgeAndGender::load_dnn_gender_classifier(std::string filename) {
	deserialize(filename) >> gender_predictor_net;
}

void AgeAndGender::load_dnn_age_predictor(std::string filename) {
	deserialize(filename) >> age_predictor_net;
}

std::vector<dlib::rectangle> AgeAndGender::from_py_list_with_tuple_to_vector_with_rectangles(py::list face_bounding_boxes) {
	std::vector<dlib::rectangle> result;
	for (auto item : face_bounding_boxes) {
		std::vector<int> face_box;
		for(py::handle num : item) {
			face_box.push_back(num.cast<int>());
		}
		
		result.push_back(
			rectangle(face_box[3], face_box[0], face_box[1], face_box[2])
		);
	}
	
	return result;
}

py::list AgeAndGender::predict(
	const py::array_t<unsigned char>& photo_numpy_array,
	py::list face_bounding_boxes = py::list()
) {
	// Load the source image.
	matrix<rgb_pixel> in = from_numpy(photo_numpy_array);
	//load_image(in, photo);

	// As proposed in the paper, use Softmax for the last layer.
	softmax<agender_type::subnet_type> gender_snet;
	gender_snet.subnet() = gender_predictor_net.subnet();	
	
	softmax<apredictor_t::subnet_type> age_snet;
	age_snet.subnet() = age_predictor_net.subnet();	

	// Evaluate the gender
	int32_t cur_face = 0;
	float confidence = 0.0f;
	enum label_ gender;
	py::list result;
	
	std::vector<dlib::rectangle> all_faces;
	if(face_bounding_boxes.size()) {
		all_faces = AgeAndGender::from_py_list_with_tuple_to_vector_with_rectangles(
			face_bounding_boxes
		);
	} else {
		all_faces = detector(in);
	}
	
	for (auto face : all_faces) {
		auto shape = sp(in, face);
		
		if (shape.num_parts()) {
			matrix<rgb_pixel> face_chip;
			
			extract_image_chip(in, get_face_chip_details(shape, 32), face_chip);
			matrix<float, 1, 2> p = mat(gender_snet(face_chip));
			
			if (p(0) < p(1)) {
				gender = male_label;
				confidence = p(1);
			} else {
				gender = female_label;
				confidence = p(0);
			}
			
			py::list face_rect;
			face_rect.append(face.left());
			face_rect.append(face.top());
			face_rect.append(face.right());
			face_rect.append(face.bottom());
			
			// Age predictor
			extract_image_chip(in, get_face_chip_details(shape, 64), face_chip);
			matrix<float, 1, number_of_age_classes> p_age = mat(age_snet(face_chip));
			
			result.append(py::dict());
			result[cur_face]["gender"] = py::dict();
			result[cur_face]["gender"]["value"] = ((gender == female_label) ? "female" : "male");
			result[cur_face]["gender"]["confidence"] = (int)floorf(confidence * 100.0f);
			
			result[cur_face]["age"] = py::dict();
			result[cur_face]["age"]["value"] = get_estimated_age(p_age, confidence);
			result[cur_face]["age"]["confidence"] = (int)floorf(confidence * 100.0f);
			result[cur_face]["face"] = face_rect;
			cur_face++;
		}
	}
	
	return result;
}

PYBIND11_MODULE(age_and_gender, m) {
	m.doc() = "Predict Age and Gender using Python";
	m.attr("__version__") = VERSION;
    py::class_<AgeAndGender>(m, "AgeAndGender")
        .def("load_shape_predictor", &AgeAndGender::load_shape_predictor)
        .def("load_dnn_gender_classifier", &AgeAndGender::load_dnn_gender_classifier)
        .def("load_dnn_age_predictor", &AgeAndGender::load_dnn_age_predictor)
        .def(
			"predict",
			&AgeAndGender::predict,
			py::arg("photo_numpy_array"),
			py::arg("face_bounding_boxes") = py::list()
		)
        .def(py::init<>());
}
