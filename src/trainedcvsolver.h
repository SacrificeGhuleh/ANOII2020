//
// Created by richard on 28.11.20.
//

#ifndef ANOII2020_TRAINEDCVSOLVER_H
#define ANOII2020_TRAINEDCVSOLVER_H

#include <opencv2/ml.hpp>

#include "trainedsolver.h"
#include "cvnetcfg.h"

class TrainedCvSolver : public TrainedSolver<CvNetCfg> {
public:
  TrainedCvSolver(const std::string &name, const std::string &fileName);

protected:
  virtual void trainImpl(const TrainInputSet &trainData, const CvNetCfg &netCfg) override;
  
  cv::Ptr<cv::ml::SVM> svm_;
  
  void process(cv::Mat &processMat);
  
  virtual void process(const cv::Mat &processMat, std::vector<float> &descriptor) = 0;
  
  void process(std::vector<cv::Mat> &processMat);

public:
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;

private:
  void convertToMl(const TrainInputSet &trainData, cv::Mat &mlTrainData, std::vector<uint8_t> &trainLabels);
};


#endif //ANOII2020_TRAINEDCVSOLVER_H
