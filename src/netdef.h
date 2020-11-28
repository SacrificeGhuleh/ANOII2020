//
// Created by richard on 24.11.20.
//

#ifndef ANOII2020_NETDEF_H
#define ANOII2020_NETDEF_H

#include <dlib/dnn.h>

// AFTER ADDING NEW DNN DEFINITION, DO NOT FORGET TO UPDATE TRAINEDDLIBSOLVER.TPP

// @formatter:off
using AlexNet = dlib::loss_multiclass_log<
  dlib::relu<dlib::fc<1000,
  dlib::dropout<dlib::relu<dlib::fc<4096,
  dlib::dropout<dlib::relu<dlib::fc<4096,
  dlib::max_pool<3, 3, 2, 2,
  dlib::relu<dlib::con<384, 3, 3, 1, 1,
  dlib::relu<dlib::con<384, 3, 3, 1, 1,
  dlib::relu<dlib::con<384, 3, 3, 1, 1,
  dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::con<256, 5, 5, 1, 1,
  dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::con<96, 11, 11, 4, 4,
  dlib::input<dlib::matrix<uint8_t>>>>>>>>>>>>>>>>>>>>>>>>;

using LeNet = dlib::loss_multiclass_log<
  dlib::fc<2,
  dlib::relu<dlib::fc<84,
  dlib::relu<dlib::fc<120,
  dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<16,5,5,1,1,
  dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<6,5,5,1,1,
  dlib::input<dlib::matrix<uint8_t>>>>>>>>>>>>>>;

using VGG19 = dlib::loss_multiclass_log<
  dlib::relu<dlib::fc<2,
  dlib::relu<dlib::fc<4096,
  dlib::dropout <
  dlib::relu<dlib::fc<4096,
  dlib::max_pool<2, 2, 2, 2,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::max_pool<2, 2, 2, 2,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::relu<dlib::con<512, 3, 3, 1, 1,
  dlib::max_pool<2, 2, 2, 2,
  dlib::relu<dlib::con<256, 3, 3, 1, 1,
  dlib::relu<dlib::con<256, 3, 3, 1, 1,
  dlib::relu<dlib::con<256, 3, 3, 1, 1,
  dlib::relu<dlib::con<256, 3, 3, 1, 1,
  dlib::max_pool<2, 2, 2, 2,
  dlib::relu<dlib::con<128, 3, 3, 1, 1,
  dlib::relu<dlib::con<128, 3, 3, 1, 1,
  dlib::max_pool<2, 2, 2, 2,
  dlib::relu<dlib::con<64, 3, 3, 1, 1,
  dlib::relu<dlib::con<64, 3, 3, 1, 1,
  dlib::input<dlib::matrix<uint8_t>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

//@formatter:on


#endif //ANOII2020_NETDEF_H
