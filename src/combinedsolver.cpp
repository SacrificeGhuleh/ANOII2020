//
// Created by richard on 06.12.20.
//

#include "combinedsolver.h"

bool CombinedSolver::detect(const cv::Mat &extractedParkingLotMat) {
  double result = 0;
  
  for (Solver *solver : solvers_) {
    result += solver->detect(extractedParkingLotMat);
  }
  result /= solvers_.size();
  return result > confidence_;
}