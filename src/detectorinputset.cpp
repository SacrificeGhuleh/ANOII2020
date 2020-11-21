//
// Created by richardzvonek on 11/21/20.
//

#include <stdexcept>
#include <fstream>
#include <random>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include "detectorinputset.h"

DetectorInputSet::DetectorInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces) {
  std::fstream test_file("data/test_images.txt");
  std::string test_path;
  
  while (test_file >> test_path) {
    //std::cout << "test_path: " << test_path << "\n";
    cv::Mat frame, gradX, gradY;
    //read testing images
    frame = cv::imread(test_path, 1);
    cv::Mat draw_frame = frame.clone();
    cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    std::vector<LoadedData> extractedSpaces;
    extractSpaces(spaces, frame, extractedSpaces);
    inputSet_.emplace_back(std::make_pair(frame, extractedSpaces));
  }
  
  std::shuffle(inputSet_.begin(), inputSet_.end(), std::mt19937(std::random_device()()));
}

const std::vector<DetectorInputSet::InputPair> &DetectorInputSet::getInputSet() const {
  return inputSet_;
}