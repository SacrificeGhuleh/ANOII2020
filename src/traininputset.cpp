//
// Created by richardzvonek on 11/21/20.
//

#include <fstream>
#include <opencv2/opencv.hpp>
#include "traininputset.h"

TrainInputSet::TrainInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces) {
  std::fstream test_file("data/test_images.txt");
  std::string test_path;
  uint8_t label;
  while (test_file >> test_path) {
    if (test_path.find("full") != std::string::npos) label = 1;
    else label = 0;
    
    //std::cout << "test_path: " << test_path << "\n";
    cv::Mat frame, gradX, gradY;
    //read testing images
    frame = cv::imread(test_path, 1);
    cv::Mat draw_frame = frame.clone();
    cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    std::vector<LoadedData> extractedSpaces;
    extractSpaces(spaces, frame, extractedSpaces);
    
    for (const LoadedData &loadedData: extractedSpaces) {
      inputSet_.emplace_back(std::make_pair(loadedData, label));
    }
  }
  
}

const std::vector<TrainInputSet::InputPair> &TrainInputSet::getInputSet() const {
  return inputSet_;
}
