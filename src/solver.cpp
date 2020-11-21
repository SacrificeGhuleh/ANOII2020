//
// Created by richardzvonek on 11/21/20.
//

#include <cassert>
#include <iostream>
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

void Solver::drawDetection() {
  int sx, sy;
  
  for (const auto &result : results) {
    cv::Mat frame = result.first->clone();
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
        cv::line(frame, cv::Point(detection.x01, detection.y01), cv::Point(detection.x03, detection.y03), Color::Black, 2);
        cv::line(frame, cv::Point(detection.x02, detection.y02), cv::Point(detection.x04, detection.y04), Color::Black, 2);
        
        cv::line(frame, cv::Point(detection.x01, detection.y01), cv::Point(detection.x02, detection.y02), Color::Black, 2);
        cv::line(frame, cv::Point(detection.x02, detection.y02), cv::Point(detection.x03, detection.y03), Color::Black, 2);
        cv::line(frame, cv::Point(detection.x03, detection.y03), cv::Point(detection.x04, detection.y04), Color::Black, 2);
        cv::line(frame, cv::Point(detection.x04, detection.y04), cv::Point(detection.x01, detection.y01), Color::Black, 2);
      } else {
        cv::circle(frame, cv::Point(sx, sy - 25), 12, Color::Black, 2);
      }
    }
    cv::imshow("Detection", frame);
    cv::waitKey();
  }
  
}
