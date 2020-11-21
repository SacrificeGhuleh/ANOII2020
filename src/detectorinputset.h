//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_DETECTORINPUTSET_H
#define ANOII2020_DETECTORINPUTSET_H


#include <vector>
#include <opencv2/core/mat.hpp>

#include "space.h"

class DetectorInputSet {
public:
  explicit DetectorInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces);
  
  
  static void loadParkingGeometry(const char *filename, std::array<Space, SPACES_COUNT> &spaces);
  
  typedef std::pair<cv::Mat, std::vector<std::pair<cv::Mat, Space>>> InputPair;
  
  const std::vector<InputPair> &getInputSet() const;

private:
  void extractSpaces(const std::array<Space, SPACES_COUNT> &spaces, const cv::Mat &inMat);
  
  std::vector<InputPair> inputSet_;
};


#endif //ANOII2020_DETECTORINPUTSET_H
