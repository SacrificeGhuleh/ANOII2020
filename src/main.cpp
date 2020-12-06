#include <iostream>
#include <fstream>

#include "detectorinputset.h"
#include "cannysolver.h"
#include "traininputset.h"
#include "traineddlibsolver.h"
#include "netdef.h"
#include "dlibnetcfg.h"
#include "hogsolver.h"

#include <cxxopts.hpp>

using AlexNetSolver = TrainedDlibSolver<AlexNet>;
using LeNetSolver = TrainedDlibSolver<LeNet>;
using Vgg19Solver = TrainedDlibSolver<VGG19>;
using ResNetSolver = TrainedDlibSolver<ResNet>;
using GoogLeNetSolver = TrainedDlibSolver<GoogLeNet>;

void getGroundTruth(const std::string &filename, std::vector<uint8_t> &groundTruthVector);

int main(int argc, char **argv) {
  cxxopts::Options options("ParkingLotOccupationDetec", "Parking lot detection tool");
  
  options.add_options()
      ("c,canny", "Use Canny detector", cxxopts::value<bool>()->default_value("false"))
      ("l,lenet", "Use LeNet", cxxopts::value<bool>()->default_value("false"))
      ("a,alex", "Use AlexNet", cxxopts::value<bool>()->default_value("false"))
      ("v,vgg19", "Use VGG19", cxxopts::value<bool>()->default_value("false"))
      ("r,resnet", "Use ResNet", cxxopts::value<bool>()->default_value("false"))
      ("g,googlenet", "Use GoogLeNet", cxxopts::value<bool>()->default_value("false"))
      ("all", "Use ALL methods", cxxopts::value<bool>()->default_value("false"))
      ("d,draw", "Draw detection", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print usage");
  
  auto cliResult = options.parse(argc, argv);
  
  if (cliResult.count("help") || argc < 2) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  
  // Load test data
  std::vector<uint8_t> groundTruth;
  getGroundTruth("data/groundtruth.txt", groundTruth);
  
  std::array<Space, SPACES_COUNT> spaces{};
  InputSet::loadParkingGeometry("data/parking_map.txt", spaces);
  
  DetectorInputSet inputSet("data/test_images.txt", spaces);
  TrainInputSet trainInputSet("data/train_images.txt", spaces);
  
  // Create instances of solvers
  CannySolver cannySolver(274, 3);
  LeNetSolver lenetSolver("LeNet", "lenet.bin", cv::Size(28, 28));
  AlexNetSolver alexNetSolver("AlexNet", "alex2.bin");
  Vgg19Solver vgg19NetSolver("VGG19", "vgg19.bin", cv::Size(32, 32));
  ResNetSolver resNetSolver("ResNet", "resnet.bin", cv::Size(32, 32));
  GoogLeNetSolver googLeNetSolver("GoogLeNet", "googlenet.bin", cv::Size(32, 32));
  
  
  if (cliResult["canny"].as<bool>() || cliResult["all"].as<bool>()) {
    cannySolver.solve(inputSet);
//    cannySolver.solve(inputSet, groundTruth); // For debugging
  }
  
  if (cliResult["lenet"].as<bool>() || cliResult["all"].as<bool>()) {
    DlibNetCfg lenetCfg(0.01, 1e-6, 128, 1000, 300);
    lenetSolver.train(trainInputSet, lenetCfg);
    lenetSolver.solve(inputSet);
  }
  
  if (cliResult["alex"].as<bool>() || cliResult["all"].as<bool>()) {
    DlibNetCfg alexNetCfg(0.01, 0.00001, 64, 128, 300);
    alexNetSolver.train(trainInputSet, alexNetCfg);
    alexNetSolver.solve(inputSet);
  }
  
  if (cliResult["vgg19"].as<bool>() || cliResult["all"].as<bool>()) {
    DlibNetCfg vgg19NetCfg(0.01, 1e-7, 64, 500, 300);
    vgg19NetSolver.train(trainInputSet, vgg19NetCfg);
    vgg19NetSolver.solve(inputSet);
  }
  
  if (cliResult["resnet"].as<bool>() || cliResult["all"].as<bool>()) {
    DlibNetCfg resNetCfg(0.01, 1e-7, 64, 128, 300);
    resNetSolver.train(trainInputSet, resNetCfg);
    resNetSolver.solve(inputSet);
  }
  
  if (cliResult["googlenet"].as<bool>() || cliResult["all"].as<bool>()) {
    DlibNetCfg googleNetCfg(0.01, 1e-5, 64, 128, 300);
    googLeNetSolver.train(trainInputSet, googleNetCfg);
    googLeNetSolver.solve(inputSet);
  }

//  CvNetCfg hogCfg;
//  HogSolver hogSolver("hog.bin");
//  hogSolver.train(trainInputSet, hogCfg);
//  hogSolver.solve(inputSet);
//  hogSolver.evaluate(groundTruth);
//  hogSolver.drawDetection();
  
  // Evaluation & drawing
  if (cliResult["canny"].as<bool>() || cliResult["all"].as<bool>()) {
    cannySolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      cannySolver.drawDetection();
    }
  }
  
  if (cliResult["lenet"].as<bool>() || cliResult["all"].as<bool>()) {
    lenetSolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      lenetSolver.drawDetection();
    }
  }
  
  if (cliResult["alex"].as<bool>() || cliResult["all"].as<bool>()) {
    alexNetSolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      alexNetSolver.drawDetection();
    }
  }
  
  if (cliResult["vgg19"].as<bool>() || cliResult["all"].as<bool>()) {
    vgg19NetSolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      vgg19NetSolver.drawDetection();
    }
  }
  
  if (cliResult["resnet"].as<bool>() || cliResult["all"].as<bool>()) {
    resNetSolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      resNetSolver.drawDetection();
    }
  }
  
  if (cliResult["googlenet"].as<bool>() || cliResult["all"].as<bool>()) {
    googLeNetSolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      googLeNetSolver.drawDetection();
    }
  }
  
  return 0;
}

void getGroundTruth(const std::string &filename, std::vector<uint8_t> &groundTruthVector) {
  std::ifstream groundTruthFile(filename);
  if (!groundTruthFile.is_open()) {
    throw std::runtime_error("Ground truth file is not opened");
  }
  
  int ground;
  while (true) {
    if (!(groundTruthFile >> ground)) break;
    groundTruthVector.emplace_back(ground);
  }
  
  groundTruthFile.close();
}



//void convert_to_ml(const std::vector<cv::Mat> &train_samples, cv::Mat &trainData) {
//  //--Convert data
//  const int rows = (int) train_samples.size();
//  const int cols = (int) std::max(train_samples[0].cols, train_samples[0].rows);
//  cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
//  trainData = cv::Mat(rows, cols, CV_32FC1);
//  std::vector<Mat>::const_iterator itr = train_samples.begin();
//  std::vector<Mat>::const_iterator end = train_samples.end();
//  for (int i = 0; itr != end; ++itr, ++i) {
//    CV_Assert(itr->cols == 1 ||
//              itr->rows == 1);
//    if (itr->cols == 1) {
//      transpose(*(itr), tmp);
//      tmp.copyTo(trainData.row(i));
//    } else if (itr->rows == 1) {
//      itr->copyTo(trainData.row(i));
//    }
//  }
//}
