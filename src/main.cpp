#include <iostream>
#include <fstream>

#include "detectorinputset.h"
#include "cannysolver.h"
#include "traininputset.h"

void getGroundTruth(const std::string &filename, std::vector<uint8_t> &groundTruthVector) {
  std::ifstream groundTruthFile(filename);
  if (!groundTruthFile.is_open()) {
    throw std::runtime_error("Ground truth file is not opened");
  }
  
  int ground;
  while (true) {
    if (!(groundTruthFile >> ground)) break;
    groundTruthVector.emplace_back(ground);
  }
  
  groundTruthFile.close();
}

int main(int argc, char **argv) {
  
  std::vector<uint8_t> groundTruth;
  getGroundTruth("data/groundtruth.txt", groundTruth);
  
  std::array<Space, SPACES_COUNT> spaces{};
  InputSet::loadParkingGeometry("data/parking_map.txt", spaces);
  
  TrainInputSet trainInputSet("data/train_images.txt", spaces);
  DetectorInputSet inputSet("data/test_images.txt", spaces);
  
  CannySolver cannySolver(274, 3);
  cannySolver.solve(inputSet);
  cannySolver.evaluate(groundTruth);
  cannySolver.drawDetection();
  
}
//void convert_to_ml(const std::vector<cv::Mat> &train_samples, cv::Mat &trainData) {
//  //--Convert data
//  const int rows = (int) train_samples.size();
//  const int cols = (int) std::max(train_samples[0].cols, train_samples[0].rows);
//  cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
//  trainData = cv::Mat(rows, cols, CV_32FC1);
//  std::vector<Mat>::const_iterator itr = train_samples.begin();
//  std::vector<Mat>::const_iterator end = train_samples.end();
//  for (int i = 0; itr != end; ++itr, ++i) {
//    CV_Assert(itr->cols == 1 ||
//              itr->rows == 1);
//    if (itr->cols == 1) {
//      transpose(*(itr), tmp);
//      tmp.copyTo(trainData.row(i));
//    } else if (itr->rows == 1) {
//      itr->copyTo(trainData.row(i));
//    }
//  }
//}
