//
// Created by richardzvonek on 11/21/20.
//

#include "trainedsolver.h"

#include <random>

void TrainedSolver::loadTrainData(std::vector<cv::Mat> &trainData) {
  
  std::shuffle(trainData.begin(), trainData.end(), std::mt19937(std::random_device()()));
}
