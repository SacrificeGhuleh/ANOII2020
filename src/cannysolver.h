//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_CANNYSOLVER_H
#define ANOII2020_CANNYSOLVER_H

#include "solver.h"

class CannySolver : public Solver {
public:
  CannySolver(uint32_t sumThreshold, uint8_t blurDiam) :
      sumThreshold_(sumThreshold),
      blurDiam_(blurDiam),
      Solver("Canny detector with threshold with OTSU") {};
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;


private:
  uint32_t sumThreshold_;
  uint8_t blurDiam_;
  
};


#endif //ANOII2020_CANNYSOLVER_H
