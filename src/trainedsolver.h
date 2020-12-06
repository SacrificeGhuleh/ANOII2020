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
  explicit TrainedSolver(const std::string &name,
                         const std::string &fileName,
                         const T_NET_CFG &netCfg)
      : filename_(fileName),
        netCfg_(netCfg),
        trained_(false),
        Solver(name) {};
  
  virtual void train(const TrainInputSet &trainData) {
    if (!trained_) {
      Timer timer;
      trainImpl(trainData);
      std::cout << "Trained in " << timer.elapsed() << " seconds\n";
      trained_ = true;
    }
  }
  
  virtual double solve(const DetectorInputSet &inputSet) override {
    if (!trained_) {
      if (!loadDnn()) {
        throw std::runtime_error("Could not load dnn bin file");
      }
    }
    return Solver::solve(inputSet);
  }
  
  virtual double solve(const DetectorInputSet &inputSet, const std::vector<uint8_t> &groundTruth) override {
    if (!trained_) {
      if (!loadDnn()) {
        throw std::runtime_error("Could not load dnn bin file");
      }
    }
    return Solver::solve(inputSet, groundTruth);
  }

protected:
  
  bool trained_;
  const std::string filename_;
  
  virtual bool loadDnn() = 0;
  
  virtual void trainImpl(const TrainInputSet &trainData) = 0;
  
  const T_NET_CFG netCfg_;
};

#endif //ANOII2020_TRAINEDSOLVER_H
