////
//// Created by richard on 28.11.20.
////
//
//#include "hogsolver.h"
//
//bool HogSolver::detect(const cv::Mat &extractedParkingLotMat) {
//  return false;
//}
#include "hogsolver.h"

HogSolver::HogSolver(const std::string &fileName) : TrainedCvSolver("HOG Solver", fileName) {
  hogDescriptor_.winSize = cv::Size(80, 80);
  hogDescriptor_.blockSize = cv::Size(8, 8);
  hogDescriptor_.cellSize = cv::Size(4, 4);
  hogDescriptor_.blockStride = cv::Size(8, 8);
  hogDescriptor_.nbins = 6;
}

void HogSolver::process(const cv::Mat &processMat, std::vector<float> &descriptor) {
  hogDescriptor_.compute(processMat, descriptor, cv::Size(4, 4), cv::Size(0, 0));
}

