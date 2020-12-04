//
// Created by richard on 28.11.20.
//

#ifndef ANOII2020_CVNETCFG_H
#define ANOII2020_CVNETCFG_H

struct CvNetCfg {
  cv::TermCriteria::Type termCriteriaType;
  uint16_t termCriteriaMaxCount;
  double termCriteriaEpsilon;
  double gamma;
  cv::ml::SVM::KernelTypes svmKernel;
};

#endif //ANOII2020_CVNETCFG_H
