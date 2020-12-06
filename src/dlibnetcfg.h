//
// Created by richard on 24.11.20.
//

#ifndef ANOII2020_DLIBNETCFG_H
#define ANOII2020_DLIBNETCFG_H

#include <cstdint>

struct DlibNetCfg {
  DlibNetCfg(float learningRate, float minLearningRate, uint16_t miniBatchSize, uint16_t stepsWithoutProgress, uint16_t maxEpochs) :
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

static DlibNetCfg lenetCfg(0.01, 1e-6, 128, 1000, 300);
static DlibNetCfg alexNetCfg(0.01, 0.00001, 64, 128, 300);
static DlibNetCfg vgg19NetCfg(0.01, 1e-7, 64, 500, 300);
static DlibNetCfg resNetCfg(0.01, 1e-7, 64, 128, 300);
static DlibNetCfg googleNetCfg(0.01, 1e-5, 64, 128, 300);


#endif //ANOII2020_DLIBNETCFG_H
