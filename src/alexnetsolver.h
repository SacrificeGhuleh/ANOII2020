//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_ALEXNETSOLVER_H
#define ANOII2020_ALEXNETSOLVER_H

#include "dlib/dnn.h"

#include "trainedsolver.h"

class AlexNetSolver : public TrainedSolver {
public:
  AlexNetSolver() : TrainedSolver("AlexNet detector") {};
  
  virtual void train(const TrainInputSet &trainData) override;
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;

private:

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
                                          dlib::input<dlib::matrix<dlib::rgb_pixel
                                          >>>>>>>>>>>>>>>>>>>>>>>>;
//@formatter:on
  
  AlexNet alexNet_;
  
};


#endif //ANOII2020_ALEXNETSOLVER_H
