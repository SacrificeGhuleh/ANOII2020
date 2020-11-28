//
// Created by richard on 28.11.20.
//


#include <filesystem>
#include <dlib/matrix.h>
#include <dlib/dnn.h>

#include <dlib/opencv/cv_image.h>
#include "traineddlibsolver.h"


template<class T_NetType>
bool TrainedDlibSolver<T_NetType>::detect(const cv::Mat &extractedParkingLotMat) {
  dlib::matrix<uint8_t> matDlib = dlib::mat(dlib::cv_image<uint8_t>(extractedParkingLotMat));
  return net_(matDlib) > 0.5f;
}

template<class T_NetType>
void TrainedDlibSolver<T_NetType>::trainImpl(const TrainInputSet &trainData, const DlibNetCfg &netCfg) {
  if (std::filesystem::exists(dnnFilename)) {
    std::cout << "Loading net from file\n";
    dlib::deserialize(dnnFilename) >> net_;
  } else {
    std::cout << "Net not found, training\n";
    
    std::vector<dlib::matrix<uint8_t>>
        trainImages;
    std::vector<unsigned long> trainLabels;
    
    for (const auto &trainElement: trainData.getInputSet()) {
      trainLabels.emplace_back(trainElement.second.occup); // label
      trainImages.emplace_back(dlib::mat(dlib::cv_image<uint8_t>(trainElement.first)));
    }
    
    dlib::dnn_trainer<T_NetType> trainer(net_, dlib::sgd());
    trainer.set_learning_rate(netCfg.learningRate);
    trainer.set_min_learning_rate(netCfg.minLearningRate);
    trainer.set_mini_batch_size(netCfg.miniBatchSize);
    trainer.set_iterations_without_progress_threshold(netCfg.stepsWithoutProgress);
    trainer.set_max_num_epochs(netCfg.maxEpochs);
    
    trainer.be_verbose();
    
    trainer.train(trainImages, trainLabels);
    
    dlib::serialize(dnnFilename) << net_;
  }
}

#include "traineddlibsolver.tpp"