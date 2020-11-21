//
// Created by richardzvonek on 11/21/20.
//

#include <opencv2/imgproc.hpp>
#include "cannysolver.h"

bool CannySolver::detect(const cv::Mat &extractedParkingLotMat) {
  cv::Mat blurImage;
  cv::Mat edgesImage;
  cv::Mat otsuImage;
  
  cv::medianBlur(extractedParkingLotMat, blurImage, blurDiam_);
  
  double otsuThreshVal = cv::threshold(blurImage, otsuImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  
  double highThreshVal = otsuThreshVal,
      lowThreshVal = otsuThreshVal * 0.5;

//      cv::Canny(blurImage, edgesImage, threshold1, threshold2);
  cv::Canny(blurImage, edgesImage, lowThreshVal, highThreshVal);
  
  uint64_t sum = 0;
  
  for (int row = 0; row < edgesImage.rows; row++) {
    for (int col = 0; col < edgesImage.cols; col++) {
      sum += edgesImage.at<cv::Vec3b>(row, col)[0] / 255;
    }
  }
  
  return sum > sumThreshold_;
}