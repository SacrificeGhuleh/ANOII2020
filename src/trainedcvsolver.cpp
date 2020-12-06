//
// Created by richard on 28.11.20.
//

#include <filesystem>
#include "trainedcvsolver.h"

TrainedCvSolver::TrainedCvSolver(
    const std::string &name,
    const std::string &fileName,
    const CvNetCfg &netCfg)
    : TrainedSolver(name, fileName, netCfg) {}

void TrainedCvSolver::trainImpl(const TrainInputSet &trainData) {
  if (std::filesystem::exists(filename_)) {
    std::cout << "Loading net from file\n";
  } else {
    std::cout << "Net not found, training\n";
    
    svm_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
    //RBF
    svm_->setGamma(0.15);
    svm_->setKernel(cv::ml::SVM::RBF);
    
    cv::Mat mlTrainData;
    std::vector<uint8_t> mlTrainLabels;
    convertToMl(trainData, mlTrainData, mlTrainLabels);
    process(mlTrainData);
    svm_->train(mlTrainData, cv::ml::ROW_SAMPLE, mlTrainLabels);
    svm_->save(filename_);
  }
}

void TrainedCvSolver::convertToMl(const TrainInputSet &trainData, cv::Mat &mlTrainData, std::vector<uint8_t> &trainLabels) {
  const int rows = (int) trainData.getInputSet().size();
  const int cols = (int) std::max(trainData.getInputSet().at(0).first.cols, trainData.getInputSet().at(0).first.rows);
  cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
  mlTrainData = cv::Mat(rows, cols, CV_32FC1);
  for (int i = 0; i < trainData.getInputSet().size(); i++) {
    CV_Assert(trainData.getInputSet().at(i).first.cols == 1 ||
              trainData.getInputSet().at(i).first.rows == 1);
    if (trainData.getInputSet().at(i).first.cols == 1) {
      transpose(trainData.getInputSet().at(i).first, tmp);
      tmp.copyTo(mlTrainData.row(i));
    } else if (trainData.getInputSet().at(i).first.rows == 1) {
      trainData.getInputSet().at(i).first.copyTo(mlTrainData.row(i));
    }
    trainLabels.emplace_back(trainData.getInputSet().at(i).second.occup);
  }
}

void TrainedCvSolver::process(cv::Mat &processMat) {
  std::vector<float> descriptors;
  process(processMat, descriptors);
  processMat = cv::Mat(descriptors).clone();
}

void TrainedCvSolver::process(std::vector<cv::Mat> &processMat) {
  for (cv::Mat &mat : processMat) {
    process(mat);
  }
}

bool TrainedCvSolver::detect(const cv::Mat &extractedParkingLotMat) {
  std::vector<float> descriptor;
  process(extractedParkingLotMat, descriptor);
  return svm_->predict(descriptor) > 0.5f;
}

