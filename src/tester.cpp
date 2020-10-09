#include <iostream>
#include <fstream>

//opencv - https://opencv.org/
#include <opencv2/opencv.hpp>

const char* winName = "Test application";

static char* matName = nullptr;

static int cannyLow = 0;
static int cannyHigh = 255;
static int medianRadius = 3;

static cv::Mat originalMat;
cv::Mat grayScaleMat;
cv::Mat blurMat;
cv::Mat cannyMat;
cv::Mat equalizedMat;



static void update(int, void *)
{
  cvtColor(originalMat, grayScaleMat, cv::COLOR_BGR2GRAY);
  
  
  cv::medianBlur(grayScaleMat, blurMat, 2*(1 + medianRadius)-1);
  cv::equalizeHist(blurMat, equalizedMat);
  cv::Canny(equalizedMat, cannyMat, cannyLow, cannyHigh);
  
  
  
  
  cvtColor(grayScaleMat, grayScaleMat, cv::COLOR_GRAY2BGR);
  cvtColor(equalizedMat, equalizedMat, cv::COLOR_GRAY2BGR);
  cvtColor(cannyMat, cannyMat, cv::COLOR_GRAY2BGR);
  
  cv::Mat newMat(cv::Size(originalMat.cols*2,originalMat.rows*2),originalMat.type(),cv::Scalar::all(0));
  cv::resizeWindow(winName, 1920.f*4.f/5.f, 1080.f*4.f/5.f);
  
  cv::Mat matRoi = newMat(cv::Rect(0,0,originalMat.cols,originalMat.rows));
  originalMat.copyTo(matRoi);
  
  matRoi = newMat(cv::Rect(originalMat.cols,0,originalMat.cols,originalMat.rows));
  grayScaleMat.copyTo(matRoi);
  
  matRoi = newMat(cv::Rect(0,originalMat.rows,originalMat.cols,originalMat.rows));
  equalizedMat.copyTo(matRoi);
  
  matRoi = newMat(cv::Rect(originalMat.cols,originalMat.rows,originalMat.cols,originalMat.rows));
  cannyMat.copyTo(matRoi);
  
  cv::imshow(winName, newMat);
}


int main(int argc, char** argv){
  if (argc < 2)
    return 1;
  
  matName = argv[1];
  
  originalMat = cv::imread(matName);
  cv::namedWindow(winName, cv::WINDOW_KEEPRATIO);
  
  cv::createTrackbar("Canny LOW", winName, &cannyLow, 255, update);
  cv::createTrackbar("Canny HIGH", winName, &cannyHigh, 255, update);
  cv::createTrackbar("Median radius", winName, &medianRadius, 15, update);
  
  update(0, nullptr);
  
  cv::waitKey();
}