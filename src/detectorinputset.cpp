//
// Created by richardzvonek on 11/21/20.
//

#include <stdexcept>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include "detectorinputset.h"

DetectorInputSet::DetectorInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces) {
  
  std::fstream test_file("data/test_images.txt");
  std::string test_path;
  
  
  while (test_file >> test_path) {
    //std::cout << "test_path: " << test_path << "\n";
    cv::Mat frame, gradX, gradY;
    //read testing images
    frame = cv::imread(test_path, 1);
    cv::Mat draw_frame = frame.clone();
    cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    extractSpaces(spaces, frame);
  }
}

void DetectorInputSet::loadParkingGeometry(const char *filename, std::array<Space, SPACES_COUNT> &spaces) {
  FILE *file = fopen(filename, "r");
  if (file == nullptr)
    throw std::runtime_error("Could not open parking geometry file");
  int ps_count, i, count;
  count = fscanf(file, "%d\n", &ps_count); // read count of polygons
  for (i = 0; i < ps_count; i++) {
    int p = 0;
    int poly_size;
    count = fscanf(file, "%d->", &poly_size); // read count of polygon vertexes
    int *row = new int[poly_size * 2];
    int j;
    for (j = 0; j < poly_size; j++) {
      int x, y;
      count = fscanf(file, "%d,%d;", &x, &y); // read vertex coordinates
      row[p] = x;
      row[p + 1] = y;
      p = p + 2;
    }
    spaces.at(i).x01 = row[0];
    spaces.at(i).y01 = row[1];
    spaces.at(i).x02 = row[2];
    spaces.at(i).y02 = row[3];
    spaces.at(i).x03 = row[4];
    spaces.at(i).y03 = row[5];
    spaces.at(i).x04 = row[6];
    spaces.at(i).y04 = row[7];
    //printf("}\n");
    free(row);
    count = fscanf(file, "\n"); // read end of line
  }
  fclose(file);
}

void DetectorInputSet::extractSpaces(const std::array<Space, SPACES_COUNT> &spaces, const cv::Mat &inMat) {
  std::vector<std::pair<cv::Mat, Space>> extractedSpaces;
  for (int x = 0; x < SPACES_COUNT; x++) {
    cv::Mat src_mat(4, 2, CV_32F);
    cv::Mat out_mat(SPACE_SIZE, CV_8U, 1);
    src_mat.at<float>(0, 0) = spaces.at(x).x01;
    src_mat.at<float>(0, 1) = spaces.at(x).y01;
    src_mat.at<float>(1, 0) = spaces.at(x).x02;
    src_mat.at<float>(1, 1) = spaces.at(x).y02;
    src_mat.at<float>(2, 0) = spaces.at(x).x03;
    src_mat.at<float>(2, 1) = spaces.at(x).y03;
    src_mat.at<float>(3, 0) = spaces.at(x).x04;
    src_mat.at<float>(3, 1) = spaces.at(x).y04;
    
    cv::Mat dest_mat(4, 2, CV_32F);
    dest_mat.at<float>(0, 0) = 0;
    dest_mat.at<float>(0, 1) = 0;
    dest_mat.at<float>(1, 0) = out_mat.cols;
    dest_mat.at<float>(1, 1) = 0;
    dest_mat.at<float>(2, 0) = out_mat.cols;
    dest_mat.at<float>(2, 1) = out_mat.rows;
    dest_mat.at<float>(3, 0) = 0;
    dest_mat.at<float>(3, 1) = out_mat.rows;
    
    cv::Mat H = cv::findHomography(src_mat, dest_mat, 0);
    warpPerspective(inMat, out_mat, H, SPACE_SIZE);
    
    extractedSpaces.emplace_back(std::make_pair(out_mat, spaces.at(x)));
  }
  inputSet_.emplace_back(std::make_pair(inMat, extractedSpaces));
}

const std::vector<DetectorInputSet::InputPair> &DetectorInputSet::getInputSet() const {
  return inputSet_;
}
