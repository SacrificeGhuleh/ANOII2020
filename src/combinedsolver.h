//
// Created by richard on 06.12.20.
//

#ifndef ANOII2020_COMBINEDSOLVER_H
#define ANOII2020_COMBINEDSOLVER_H

#include <vector>

#include "solver.h"


class CombinedSolver : public Solver {
public:
  CombinedSolver(const std::string &name, const std::vector<Solver *> &solvers, double confidence = 0.5) : confidence_(confidence),
                                                                                                           solvers_(solvers),
                                                                                                           Solver(name) {};
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;

private:
  const double confidence_;
  const std::vector<Solver *> solvers_;
};


#endif //ANOII2020_COMBINEDSOLVER_H
