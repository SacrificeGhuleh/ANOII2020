//
// Created by richardzvonek on 11/21/20.
//

#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>
#include "traininputset.h"
#include "colors.h"

TrainInputSet::TrainInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces) {
  std::fstream test_file(filename);
  std::string test_path;
  uint8_t label;
  while (test_file >> test_path) {
    if (test_path.find("full") != std::string::npos) label = 1;
    else label = 0;

//    std::cout << test_path << " label: " << label << "\n";
    
    //std::cout << "test_path: " << test_path << "\n";
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
    }
  }
  
  std::shuffle(inputSet_.begin(), inputSet_.end(), std::mt19937(std::random_device()()));

//  for (const auto &elem : inputSet_) {
//    auto color = Color::Green;
//    if (elem.second.occup != 0) {
//      color = Color::Red;
//    }
//
//    cv::Mat displayMat;
//    cv::cvtColor(elem.first, displayMat, cv::COLOR_GRAY2BGR);
//
//    std::stringstream debugText1;
//    debugText1 << static_cast<int>(elem.second.occup);
//    cv::putText(displayMat, debugText1.str(), cv::Point(40, 40), cv::FONT_HERSHEY_PLAIN, 2, color);
//
//    cv::imshow("Train", displayMat);
//    cv::waitKey(100);
//  }
//  cv::destroyAllWindows();
}

const std::vector<InputSet::LoadedData> &TrainInputSet::getInputSet() const {
  return inputSet_;
}
