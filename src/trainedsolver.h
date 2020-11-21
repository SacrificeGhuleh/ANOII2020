//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_TRAINEDSOLVER_H
#define ANOII2020_TRAINEDSOLVER_H

#include "solver.h"

class TrainedSolver : public Solver {
public:
  explicit TrainedSolver(const std::string &name) : Solver(name) {};
  
  typedef std::pair<cv::Mat, uint8_t> TrainPair;
  
  virtual void train(const std::vector<cv::Mat> &trainData) = 0;
  
  
  static void loadTrainData(std::vector<cv::Mat> &trainData);
};


#endif //ANOII2020_TRAINEDSOLVER_H
