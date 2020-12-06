//
// Created by richard on 28.11.20.
//


#include <filesystem>
#include <dlib/matrix.h>
#include <dlib/dnn.h>

#include <dlib/opencv/cv_image.h>
#include <opencv2/imgproc.hpp>
#include "traineddlibsolver.h"


template<class T_NetType>
bool TrainedDlibSolver<T_NetType>::detect(const cv::Mat &extractedParkingLotMat) {
//  cv::Mat dnnInput = extractedParkingLotMat.clone();
//  resize(dnnInput);
  cv::Mat dnnInput;
  cv::resize(extractedParkingLotMat, dnnInput, resizeTo_);
  dlib::matrix<uint8_t> matDlib = dlib::mat(dlib::cv_image<uint8_t>(dnnInput));
  return net_(matDlib) > 0.5f;
}

template<class T_NetType>
void TrainedDlibSolver<T_NetType>::trainImpl(const TrainInputSet &trainData) {
  if (!loadDnn()) {
    std::cout << "Net not found, training\n";
    
    std::vector<dlib::matrix<uint8_t>>
        trainImages;
    std::vector<unsigned long> trainLabels;
    
    for (const auto &trainElement: trainData.getInputSet()) {
      trainLabels.emplace_back(trainElement.second.occup); // label
      //cv::Mat dnnInput = trainElement.first.clone();
      //resize(dnnInput);
      cv::Mat dnnInput;
      cv::resize(trainElement.first, dnnInput, resizeTo_);
      trainImages.emplace_back(dlib::mat(dlib::cv_image<uint8_t>(dnnInput)));
    }
    
    dlib::dnn_trainer<T_NetType> trainer(net_, dlib::sgd());
    trainer.set_learning_rate(netCfg_.learningRate);
    trainer.set_min_learning_rate(netCfg_.minLearningRate);
    trainer.set_mini_batch_size(netCfg_.miniBatchSize);
    trainer.set_iterations_without_progress_threshold(netCfg_.stepsWithoutProgress);
    trainer.set_max_num_epochs(netCfg_.maxEpochs);
    
    trainer.be_verbose();
    
    trainer.train(trainImages, trainLabels);
    net_.clean();
    
    dlib::serialize(filename_) << net_;
  }
}

template<class T_NetType>
void TrainedDlibSolver<T_NetType>::resize(cv::Mat &resizeMat) {
  if (resizeTo_ != cv::Size(80, 80)) {
    cv::resize(resizeMat, resizeMat, resizeTo_);
  }
}

template<class T_NetType>
bool TrainedDlibSolver<T_NetType>::loadDnn() {
  if (std::filesystem::exists(filename_)) {
    std::cout << "Loading net from file\n";
    dlib::deserialize(filename_) >> net_;
    return true;
  }
  return false;
}

#include "traineddlibsolver.tpp"