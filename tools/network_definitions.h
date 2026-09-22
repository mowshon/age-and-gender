#ifndef AGE_AND_GENDER_TOOLS_NETWORK_DEFINITIONS_H_
#define AGE_AND_GENDER_TOOLS_NETWORK_DEFINITIONS_H_

#include <dlib/dnn.h>

namespace age_gender_models {

using namespace dlib;

constexpr unsigned long number_of_age_classes = 81;

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
using age_network = loss_multiclass_log<
    fc<number_of_age_classes, aresnet10_backbone<input_rgb_image>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using gender_block = BN<con<N, 3, 3, stride, stride,
    relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET>
using gender_affine_block = relu<gender_block<N, affine, 1, SUBNET>>;
template <typename SUBNET>
using gender_level1 = avg_pool<2, 2, 2, 2, gender_affine_block<64, SUBNET>>;
template <typename SUBNET>
using gender_level2 = avg_pool<2, 2, 2, 2, gender_affine_block<32, SUBNET>>;
using gender_network = loss_multiclass_log<fc<2, multiply<relu<fc<16,
    multiply<gender_level1<gender_level2<input_rgb_image_sized<32>>>>>>>>>;

}  // namespace age_gender_models

#endif  // AGE_AND_GENDER_TOOLS_NETWORK_DEFINITIONS_H_
