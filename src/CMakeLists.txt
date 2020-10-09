CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(ParkingLotOccupationDetector)

# add libs you need
SET(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)
SET(SRC main.cpp)

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC} cli.h)
ADD_EXECUTABLE(CannyTester tester.cpp cli.h)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(CannyTester PRIVATE ${OpenCV_LIBS})