//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_TRAINEDSOLVER_H
#define ANOII2020_TRAINEDSOLVER_H

#include "traininputset.h"
#include "solver.h"

class TrainedSolver : public Solver {
public:
  TrainedSolver(const std::string &name);
  
  virtual void train(const TrainInputSet &trainData);
private:
  virtual void trainImpl(const TrainInputSet &trainData) = 0;
};


#endif //ANOII2020_TRAINEDSOLVER_H
