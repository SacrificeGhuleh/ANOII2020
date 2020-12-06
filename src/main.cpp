#include <iostream>
#include <fstream>

#include "detectorinputset.h"
#include "cannysolver.h"
#include "traininputset.h"
#include "traineddlibsolver.h"
#include "netdef.h"
#include "dlibnetcfg.h"
#include "hogsolver.h"
#include "combinedsolver.h"

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
      ("comb1", "Combination of AlexNet, ResNet and GoogLeNet", cxxopts::value<bool>()->default_value("false"))
      ("combultimate", "Combination of ALL solvers", cxxopts::value<bool>()->default_value("false"))
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
  LeNetSolver lenetSolver("LeNet", "lenet.bin", lenetCfg, cv::Size(28, 28));
  AlexNetSolver alexNetSolver("AlexNet", "alex2.bin", alexNetCfg);
  Vgg19Solver vgg19NetSolver("VGG19", "vgg19.bin", vgg19NetCfg, cv::Size(32, 32));
  ResNetSolver resNetSolver("ResNet", "resnet.bin", resNetCfg, cv::Size(32, 32));
  GoogLeNetSolver googLeNetSolver("GoogLeNet", "googlenet.bin", googleNetCfg, cv::Size(32, 32));
  
  CombinedSolver comb1Solver("Combination of AlexNet, ResNet and GoogLeNet", std::vector<Solver *>{
      &alexNetSolver,
      &resNetSolver,
      &googLeNetSolver});
  
  CombinedSolver ultimateSolver("Ultimate crazy combination of all methods", std::vector<Solver *>{
      &cannySolver,
      &lenetSolver,
      &alexNetSolver,
      &vgg19NetSolver,
      &resNetSolver}, 4. / 5.);
  
  
  if (cliResult["canny"].as<bool>() || cliResult["all"].as<bool>()) {
    cannySolver.solve(inputSet);
//    cannySolver.solve(inputSet, groundTruth); // For debugging
  }
  
  if (cliResult["lenet"].as<bool>() || cliResult["all"].as<bool>()) {
    lenetSolver.train(trainInputSet);
    lenetSolver.solve(inputSet);
  }
  
  if (cliResult["alex"].as<bool>() || cliResult["all"].as<bool>()) {
    alexNetSolver.train(trainInputSet);
    alexNetSolver.solve(inputSet);
  }
  
  if (cliResult["vgg19"].as<bool>() || cliResult["all"].as<bool>()) {
    vgg19NetSolver.train(trainInputSet);
    vgg19NetSolver.solve(inputSet);
  }
  
  if (cliResult["resnet"].as<bool>() || cliResult["all"].as<bool>()) {
    resNetSolver.train(trainInputSet);
    resNetSolver.solve(inputSet);
  }
  
  if (cliResult["googlenet"].as<bool>() || cliResult["all"].as<bool>()) {
    googLeNetSolver.train(trainInputSet);
    googLeNetSolver.solve(inputSet);
  }
  
  if (cliResult["comb1"].as<bool>() || cliResult["all"].as<bool>()) {
    comb1Solver.solve(inputSet);
  }
  
  if (cliResult["combultimate"].as<bool>() || cliResult["all"].as<bool>()) {
    ultimateSolver.solve(inputSet);
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
  
  if (cliResult["comb1"].as<bool>() || cliResult["all"].as<bool>()) {
    comb1Solver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      comb1Solver.drawDetection();
    }
  }
  
  if (cliResult["combultimate"].as<bool>() || cliResult["all"].as<bool>()) {
    ultimateSolver.evaluate(groundTruth);
    if (cliResult["draw"].as<bool>()) {
      ultimateSolver.drawDetection();
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