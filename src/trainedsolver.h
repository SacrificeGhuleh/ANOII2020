//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_TRAINEDSOLVER_H
#define ANOII2020_TRAINEDSOLVER_H

#include "traininputset.h"
#include "solver.h"
#include "netcfg.h"

template<class T_NetType>
class TrainedSolver : public Solver {
public:
  TrainedSolver(const std::string &name, const std::string &fileName) : Solver(name), dnnFilename(fileName) {};
  
  virtual void train(const TrainInputSet &trainData, const NetCfg &netCfg);
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;

protected:
  
  virtual void trainImpl(const TrainInputSet &trainData, const NetCfg &netCfg);
  
  const std::string dnnFilename;
  T_NetType net_;
};

#endif //ANOII2020_TRAINEDSOLVER_H
