//
// Created by richardzvonek on 11/21/20.
//

#include "alexnetsolver.h"

#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

void AlexNetSolver::trainImpl(const TrainInputSet &trainData) {
  std::vector<dlib::matrix<uint8_t>> trainImages;
  std::vector<unsigned long> trainLabels;
  
  for (const auto &trainElement: trainData.getInputSet()) {
    trainLabels.emplace_back(trainElement.second.occup); // label
    trainImages.emplace_back(dlib::mat(dlib::cv_image<uint8_t>(trainElement.first)));
  }
  
  dlib::dnn_trainer<AlexNet> trainer(alexNet_, dlib::sgd());
  trainer.set_learning_rate(0.01);
  trainer.set_min_learning_rate(0.001);
  trainer.set_mini_batch_size(256);
  trainer.set_iterations_without_progress_threshold(1000);
  trainer.set_max_num_epochs(300);
  
  trainer.be_verbose();
  
  trainer.train(trainImages, trainLabels);
}

bool AlexNetSolver::detect(const cv::Mat &extractedParkingLotMat) {
  dlib::matrix<uint8_t> matDlib = dlib::mat(dlib::cv_image<uint8_t>(extractedParkingLotMat));
  return alexNet_(matDlib) > 0.5f;
}
