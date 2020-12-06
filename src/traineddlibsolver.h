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
  TrainedDlibSolver(const std::string &name,
                    const std::string &fileName,
                    const DlibNetCfg &netCfg,
                    const cv::Size resizeTo = cv::Size(80, 80))
      : resizeTo_(resizeTo),
        TrainedSolver(name, fileName, netCfg) {};

//  virtual void train(const TrainInputSet &trainData, const DlibNetCfg &netCfg);
  
  virtual bool detect(const cv::Mat &extractedParkingLotMat) override;

protected:
  void resize(cv::Mat &resizeMat);
  
  virtual void trainImpl(const TrainInputSet &trainData);
  
  virtual bool loadDnn() override;
  
  T_NetType net_;
  const cv::Size resizeTo_;
};


#endif //ANOII2020_TRAINEDDLIBSOLVER_H
