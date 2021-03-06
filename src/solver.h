//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_SOLVER_H
#define ANOII2020_SOLVER_H

// STL headers
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

// OpenCV headers
#include <opencv2/core/mat.hpp>

// My headers
#include "solvescore.h"
#include "detectorinputset.h"

class Solver {
public:
  explicit Solver(const std::string &name) : solverName(name) {};
  
  SolveScore evaluate(const std::vector<uint8_t> &groundTruth);
  
  virtual double solve(const DetectorInputSet &inputSet);
  
  virtual double solve(const DetectorInputSet &inputSet, const std::vector<uint8_t> &groundTruth);
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) = 0;
  
  void drawDetection(int delay = 0, bool saveToDisk = false, const std::filesystem::path &path = std::filesystem::path("out/"));

protected:
  typedef std::pair<const cv::Mat *, std::vector<Space>> OccupancyData;
  const std::string solverName;
  std::vector<OccupancyData> results;
  
};


#endif //ANOII2020_SOLVER_H
