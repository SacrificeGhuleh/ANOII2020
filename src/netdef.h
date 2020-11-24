//
// Created by richard on 24.11.20.
//

#ifndef ANOII2020_NETDEF_H
#define ANOII2020_NETDEF_H
// @formatter:off

#include <dlib/dnn.h>

// AFTER ADDING NEW DNN DEFINITION, DO NOT FORGET TO UPDATE TRAINEDSOLVER.TPP

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
    dlib::input<dlib::matrix<uint8_t
>>>>>>>>>>>>>>>>>>>>>>>>;
//@formatter:on


#endif //ANOII2020_NETDEF_H
