//
// Created by richard on 24.11.20.
//

#ifndef ANOII2020_NETDEF_H
#define ANOII2020_NETDEF_H

#include <dlib/dnn.h>

// AFTER ADDING NEW DNN DEFINITION, DO NOT FORGET TO UPDATE TRAINEDDLIBSOLVER.TPP

const uint8_t numberOfClasses = 2;

// @formatter:off

// http://dlib.net/dnn_introduction_ex.cpp.html
using LeNet = dlib::loss_multiclass_log<
  dlib::fc<numberOfClasses,
  dlib::relu<dlib::fc<84,
  dlib::relu<dlib::fc<120,
  dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<16,5,5,1,1,
  dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<6,5,5,1,1,
  dlib::input<dlib::matrix<uint8_t>>>>>>>>>>>>>>;

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

using VGG19 = dlib::loss_multiclass_log<
  dlib::relu<dlib::fc<numberOfClasses,
  dlib::relu<dlib::fc<4096,
  dlib::dropout<dlib::relu<dlib::fc<4096,
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

// http://dlib.net/dnn_introduction2_ex.cpp.html
template <
    int N,
    template <typename> class BN,
    int stride,
    typename SUBNET
>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <
    template <int,template<typename>class,int,typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET
>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <
    template <int,template<typename>class,int,typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET
>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;


template <typename SUBNET> using res       = dlib::relu<residual<block,8,dlib::bn_con,SUBNET>>;

template <typename SUBNET> using res_down  = dlib::relu<residual_down<block,8,dlib::bn_con,SUBNET>>;

using ResNet = dlib::loss_multiclass_log<dlib::fc<numberOfClasses,
    dlib::avg_pool_everything<
    res<res<res<res_down<
    dlib::repeat<9,res, // repeat this layer 9 times
    res_down<
    res<
    dlib::input<dlib::matrix<uint8_t>>>>>>>>>>>>;


//http://dlib.net/dnn_inception_ex.cpp.html
// Inception layer has some different convolutions inside.  Here we define
// blocks as convolutions with different kernel size that we will use in
// inception layer block.
template <typename SUBNET> using block_a1 = dlib::relu<dlib::con<10,1,1,1,1,SUBNET>>;
template <typename SUBNET> using block_a2 = dlib::relu<dlib::con<10,3,3,1,1,dlib::relu<dlib::con<16,1,1,1,1,SUBNET>>>>;
template <typename SUBNET> using block_a3 = dlib::relu<dlib::con<10,5,5,1,1,dlib::relu<dlib::con<16,1,1,1,1,SUBNET>>>>;
template <typename SUBNET> using block_a4 = dlib::relu<dlib::con<10,1,1,1,1,dlib::max_pool<3,3,1,1,SUBNET>>>;

// Here is inception layer definition. It uses different blocks to process input
// and returns combined output.  Dlib includes a number of these inceptionN
// layer types which are themselves created using concat layers.
template <typename SUBNET> using incept_a = dlib::inception4<block_a1,block_a2,block_a3,block_a4, SUBNET>;

// Network can have inception layers of different structure.  It will work
// properly so long as all the sub-blocks inside a particular inception block
// output tensors with the same number of rows and columns.
template <typename SUBNET> using block_b1 = dlib::relu<dlib::con<4,1,1,1,1,SUBNET>>;
template <typename SUBNET> using block_b2 = dlib::relu<dlib::con<4,3,3,1,1,SUBNET>>;
template <typename SUBNET> using block_b3 = dlib::relu<dlib::con<4,1,1,1,1,dlib::max_pool<3,3,1,1,SUBNET>>>;
template <typename SUBNET> using incept_b = dlib::inception3<block_b1,block_b2,block_b3,SUBNET>;

// Now we can define a simple network for classifying MNIST digits.  We will
// train and test this network in the code below.
using GoogLeNet = dlib::loss_multiclass_log<
  dlib::fc<numberOfClasses,
  dlib::relu<dlib::fc<32,
  dlib::max_pool<2,2,2,2,incept_b<
  dlib::max_pool<2,2,2,2,incept_a<
  dlib::input<dlib::matrix<uint8_t>>
>>>>>>>>;
//@formatter:on


#endif //ANOII2020_NETDEF_H
