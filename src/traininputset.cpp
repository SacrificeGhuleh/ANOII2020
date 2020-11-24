//
// Created by richardzvonek on 11/21/20.
//

#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>
#include "traininputset.h"
#include "colors.h"

TrainInputSet::TrainInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces) {
  std::cout << "Creating input for DNNs\n";
  std::fstream test_file(filename);
  std::string test_path;
  uint8_t label;
  while (test_file >> test_path) {
    if (test_path.find("full") != std::string::npos) label = 1;
    else label = 0;
    
    cv::Mat frame, gradX, gradY;
    //read testing images
    frame = cv::imread(test_path, 1);
    
    cv::Mat draw_frame = frame.clone();
    cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    std::vector<LoadedData> extractedSpaces;
    extractSpaces(spaces, frame, extractedSpaces);
    
    for (LoadedData &loadedData: extractedSpaces) {
      loadedData.second.occup = label;
      inputSet_.emplace_back(loadedData);
      
      std::vector<cv::Mat> extended;
      getExtendedImages(loadedData.first, extended);
      for (const cv::Mat &extImg : extended) {
        inputSet_.emplace_back(std::make_pair(extImg, loadedData.second));
      }
    }
  }
  
  std::shuffle(inputSet_.begin(), inputSet_.end(), std::mt19937(std::random_device()()));
  std::cout << "Input for DNNs created\n";
}

const std::vector<InputSet::LoadedData> &TrainInputSet::getInputSet() const {
  return inputSet_;
}

void TrainInputSet::getExtendedImages(const cv::Mat &inputImg, std::vector<cv::Mat> &extendedImages) {
  cv::Mat flippedx;
  cv::Mat flippedy;
  cv::Mat flippedxy;
  
  cv::flip(inputImg, flippedx, 0);
  cv::flip(inputImg, flippedy, 1);
  cv::flip(inputImg, flippedxy, -1);
  
  extendedImages.emplace_back(flippedx);
  extendedImages.emplace_back(flippedy);
  extendedImages.emplace_back(flippedxy);
  
  // Simulate night
  extendedImages.emplace_back(inputImg / 2);
  extendedImages.emplace_back(flippedx / 3);
  extendedImages.emplace_back(flippedy / 4);
  extendedImages.emplace_back(flippedxy / 5);
}
