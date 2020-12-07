//
// Created by richardzvonek on 11/21/20.
//

#include <cassert>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "solver.h"
#include "timer.h"
#include "colors.h"

SolveScore Solver::evaluate(const std::vector<uint8_t> &groundTruth) {
  assert(!groundTruth.empty());
  assert(!results.empty());

//  assert(groundTruth.size() == results.size());
  
  int falsePositives = 0;
  int falseNegatives = 0;
  int truePositives = 0;
  int trueNegatives = 0;
  uint8_t ground, detect;
  
  int i = 0;
  for (const auto &result : results) {
    for (const auto &detection : result.second) {
      ground = groundTruth.at(i);
      detect = detection.occup;
      
      //false positives
      if ((detect == 1) && (ground == 0)) {
        falsePositives++;
      }
      
      //false negatives
      if ((detect == 0) && (ground == 1)) {
        falseNegatives++;
      }
      
      //true positives
      if ((detect == 1) && (ground == 1)) {
        truePositives++;
      }
      
      //true negatives
      if ((detect == 0) && (ground == 0)) {
        trueNegatives++;
      }
      
      i++;
    }
  }
  
  SolveScore solveScore(falsePositives, falseNegatives, truePositives, trueNegatives);
  
  std::cout << solverName
            << "\n - falsePositives " << solveScore.getFalsePositives()
            << "\n - falseNegatives " << solveScore.getFalseNegatives()
            << "\n - truePositives " << solveScore.getTruePositives()
            << "\n - trueNegatives " << solveScore.getTrueNegatives()
            << "\n - Accuracy    " << solveScore.getAccuracy()
            << "\n - Sensitivity " << solveScore.getSensitivity()
            << "\n - f1 score    " << solveScore.getF1Score()
            << "\n------------------------\n";
  
  return solveScore;
}

double Solver::solve(const DetectorInputSet &inputSet) {
  Timer timer;
  const auto &inputPairVector = inputSet.getInputSet();
  for (const auto &inputPair : inputPairVector) {
    Solver::OccupancyData occupancyData;
    
    for (const auto &extractedParkingLot : inputPair.second) {
      Space space = extractedParkingLot.second;
      space.occup = detect(extractedParkingLot.first);
      occupancyData.second.emplace_back(space);
    }
    occupancyData.first = &inputPair.first;
    results.emplace_back(occupancyData);
  }
  double elapsed = timer.elapsed();
  std::cout << solverName << " solved in " << elapsed << " seconds\n";
  return elapsed;
}


double Solver::solve(const DetectorInputSet &inputSet, const std::vector<uint8_t> &groundTruth) {
  Timer timer;
  const auto &inputPairVector = inputSet.getInputSet();
  int i = 0;
  for (const auto &inputPair : inputPairVector) {
    Solver::OccupancyData occupancyData;
    cv::imshow("Current image", inputPair.first);
    
    for (const auto &extractedParkingLot : inputPair.second) {
      Space space = extractedParkingLot.second;
      space.occup = detect(extractedParkingLot.first);
      auto color = Color::Green;
      if (space.occup != groundTruth.at(i)) {
        color = Color::Red;
      }
      
      cv::Mat viewMat;
      cv::cvtColor(extractedParkingLot.first, viewMat, cv::COLOR_GRAY2BGR);
      std::stringstream debugText1;
      debugText1 << "Det: " << space.occup;
      cv::putText(viewMat, debugText1.str(), cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, color);
      
      std::stringstream debugText2;
      debugText2 << "Gnd: " << static_cast<int>(groundTruth.at(i));
      cv::putText(viewMat, debugText2.str(), cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1, color);
      
      cv::imshow("Wrong detection", viewMat);
      cv::waitKey(0);
      
      occupancyData.second.emplace_back(space);
      i++;
    }
    occupancyData.first = &inputPair.first;
    results.emplace_back(occupancyData);
  }
  double elapsed = timer.elapsed();
  std::cout << solverName << " solved in " << elapsed << " seconds\n";
  cv::destroyAllWindows();
  return elapsed;
}


void Solver::drawDetection(int delay, bool saveToDisk, const std::filesystem::path &path) {
  int sx, sy;
  int counter = 0;
  
  
  if (saveToDisk) {
    if (!std::filesystem::is_directory(path) || !std::filesystem::exists(path)) { // Check if src folder exists
      std::filesystem::create_directories(path); // create src folder
    }
  }
  
  for (const auto &result : results) {
    cv::Mat frame = result.first->clone();
    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    for (const auto &detection : result.second) {
      cv::Point pt1, pt2;
      pt1.x = detection.x01;
      pt1.y = detection.y01;
      pt2.x = detection.x03;
      pt2.y = detection.y03;
      sx = (pt1.x + pt2.x) / 2;
      sy = (pt1.y + pt2.y) / 2;
      if (detection.occup) {
//        cv::circle(frame, cv::Point(sx, sy - 25), 12, Color::Black, -1);
        cv::line(frame, cv::Point(detection.x01, detection.y01), cv::Point(detection.x03, detection.y03), Color::Red, 2);
        cv::line(frame, cv::Point(detection.x02, detection.y02), cv::Point(detection.x04, detection.y04), Color::Red, 2);
        
        cv::line(frame, cv::Point(detection.x01, detection.y01), cv::Point(detection.x02, detection.y02), Color::Red, 2);
        cv::line(frame, cv::Point(detection.x02, detection.y02), cv::Point(detection.x03, detection.y03), Color::Red, 2);
        cv::line(frame, cv::Point(detection.x03, detection.y03), cv::Point(detection.x04, detection.y04), Color::Red, 2);
        cv::line(frame, cv::Point(detection.x04, detection.y04), cv::Point(detection.x01, detection.y01), Color::Red, 2);
      } else {
        cv::circle(frame, cv::Point(sx, sy - 25), 12, Color::Green, 2);
      }
    }
    cv::putText(frame, solverName, cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2, Color::Red, 2);
    cv::imshow("Detection", frame);
    if (saveToDisk) {
      std::stringstream ss;
      ss << "detection" << counter << ".png";
      
      std::filesystem::path writePath = path;
      writePath.append(ss.str());
      cv::imwrite(writePath.string(), frame);
    }
    counter++;
    cv::waitKey(delay);
  }
  cv::destroyAllWindows();
}
