//
// Created by richard on 28.11.20.
//

#ifndef ANOII2020_TRAINEDDLIBSOLVER_H
#define ANOII2020_TRAINEDDLIBSOLVER_H

#include "trainedsolver.h"
#include "dlibnetcfg.h"

template<class T_NetType>
class TrainedDlibSolver : public TrainedSolver<DlibNetCfg> {
public:
  TrainedDlibSolver(const std::string &name, const std::string &fileName) : TrainedSolver(name), dnnFilename(fileName) {};

//  virtual void train(const TrainInputSet &trainData, const DlibNetCfg &netCfg);
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;

protected:
  
  virtual void trainImpl(const TrainInputSet &trainData, const DlibNetCfg &netCfg);
  
  const std::string dnnFilename;
  T_NetType net_;
};


#endif //ANOII2020_TRAINEDDLIBSOLVER_H
