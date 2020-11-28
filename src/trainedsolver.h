//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_TRAINEDSOLVER_H
#define ANOII2020_TRAINEDSOLVER_H

#include <iostream>

#include "traininputset.h"
#include "solver.h"
#include "timer.h"

template<typename T_NET_CFG>
class TrainedSolver : public Solver {
public:
  explicit TrainedSolver(const std::string &name) : Solver(name) {};
  
  virtual void train(const TrainInputSet &trainData, const T_NET_CFG &netCfg) {
    Timer timer;
    trainImpl(trainData, netCfg);
    std::cout << "Trained in " << timer.elapsed() << " seconds\n";
  }

protected:
  
  virtual void trainImpl(const TrainInputSet &trainData, const T_NET_CFG &netCfg) = 0;
};

#endif //ANOII2020_TRAINEDSOLVER_H
