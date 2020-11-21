cmake_minimum_required(VERSION 3.14)
project(ParkingLotOccupationDetector)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)
set(SRC main.cpp timer.h colors.h solver.cpp solver.h trainedsolver.cpp trainedsolver.h solvescore.h cannysolver.cpp cannysolver.h detectorinputset.cpp detectorinputset.h space.h inputset.cpp inputset.h traininputset.cpp traininputset.h alexnetsolver.cpp alexnetsolver.h)

add_executable(${PROJECT_NAME} ${SRC} cli.h)
add_executable(CannyTester tester.cpp cli.h)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} dlib::dlib)
target_link_libraries(CannyTester PRIVATE ${OpenCV_LIBS})