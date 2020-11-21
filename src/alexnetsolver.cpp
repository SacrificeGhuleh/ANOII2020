//
// Created by richardzvonek on 11/21/20.
//

#include "alexnetsolver.h"

#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

void AlexNetSolver::train(const TrainInputSet &trainData) {
  std::vector<dlib::matrix<dlib::rgb_pixel>> trainImages;
  std::vector<unsigned long> trainLabels;
  
  for (const auto &trainElement: trainData.getInputSet()) {
    trainLabels.emplace_back(trainElement.second); // label
    
    const cv::Mat &trainImg = trainElement.first.first;
    
    trainImages.emplace_back(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(trainImg)));
  }
  
  dlib::dnn_trainer<AlexNet> trainer(alexNet_, dlib::sgd(), {0});
  trainer.set_learning_rate(1e-5);
  trainer.set_min_learning_rate(1e-8);
  trainer.set_mini_batch_size(256);
  trainer.set_iterations_without_progress_threshold(1000);
  trainer.set_max_num_epochs(200);
  
  trainer.be_verbose();
  
  trainer.train(trainImages, trainLabels);
}

bool AlexNetSolver::detect(const cv::Mat &extractedParkingLotMat) {
  dlib::matrix<dlib::rgb_pixel> matDlib = dlib::mat(dlib::cv_image<dlib::rgb_pixel>(extractedParkingLotMat));
  return alexNet_(matDlib);
}
