//
// Created by richardzvonek on 11/21/20.
//

#include <iostream>
#include "trainedsolver.h"
#include "timer.h"

TrainedSolver::TrainedSolver(const std::string &name) : Solver(name) {}

void TrainedSolver::train(const TrainInputSet &trainData) {
  Timer timer;
  trainImpl(trainData);
  std::cout << "Trained in " << timer.elapsed() << " seconds\n";
}
