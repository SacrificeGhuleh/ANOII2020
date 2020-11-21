//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_INPUTSET_H
#define ANOII2020_INPUTSET_H

#include <array>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "space.h"


class InputSet {
public:
  typedef std::pair<cv::Mat, Space> LoadedData;
  
  static void extractSpaces(const std::array<Space, SPACES_COUNT> &spaces, const cv::Mat &inMat, std::vector<LoadedData> &extractedSpaces);
  
  static void loadParkingGeometry(const char *filename, std::array<Space, SPACES_COUNT> &spaces);
};


#endif //ANOII2020_INPUTSET_H
