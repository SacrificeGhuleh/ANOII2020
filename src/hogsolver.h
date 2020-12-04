//
// Created by richard on 28.11.20.
//

#ifndef ANOII2020_HOGSOLVER_H
#define ANOII2020_HOGSOLVER_H

#include <opencv2/objdetect.hpp>
#include "trainedcvsolver.h"

class HogSolver : public TrainedCvSolver {
//public:
//  HogSolver() : TrainedCvSolver("HOG solver") {
//    hogDescriptor_.winSize = cv::Size(80, 80);
//  }
//
//  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;
//
public:
  HogSolver(const std::string &fileName);

protected:
  virtual void process(const cv::Mat &processMat, std::vector<float> &descriptor) override;

private:
  cv::HOGDescriptor hogDescriptor_;
};


#endif //ANOII2020_HOGSOLVER_H
