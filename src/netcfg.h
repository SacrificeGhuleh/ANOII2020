//
// Created by richard on 24.11.20.
//

#ifndef ANOII2020_NETCFG_H
#define ANOII2020_NETCFG_H

#include <cstdint>

struct NetCfg {
  NetCfg(float learningRate, float minLearningRate, uint16_t miniBatchSize, uint16_t stepsWithoutProgress, uint16_t maxEpochs) :
      learningRate(learningRate),
      minLearningRate(minLearningRate),
      miniBatchSize(miniBatchSize),
      stepsWithoutProgress(stepsWithoutProgress),
      maxEpochs(maxEpochs) {}
  
  float learningRate;
  float minLearningRate;
  uint16_t miniBatchSize;
  uint16_t stepsWithoutProgress;
  uint16_t maxEpochs;
};

#endif //ANOII2020_NETCFG_H
