//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_TRAINEDSOLVER_H
#define ANOII2020_TRAINEDSOLVER_H

#include "traininputset.h"
#include "solver.h"

class TrainedSolver : public Solver {
public:
  TrainedSolver(const std::string &name, const std::string &fileName) : Solver(name), dnnFilename(fileName) {};
  
  virtual void train(const TrainInputSet &trainData);

protected:
  const std::string dnnFilename;

private:
  virtual void trainImpl(const TrainInputSet &trainData) = 0;
};


#endif //ANOII2020_TRAINEDSOLVER_H
