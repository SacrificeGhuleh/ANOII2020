cmake_minimum_required(VERSION 3.14)
project(ParkingLotOccupationDetector)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d opencv_objdetect)
set(SRC main.cpp timer.h colors.h solver.cpp solver.h trainedsolver.h solvescore.h cannysolver.cpp cannysolver.h detectorinputset.cpp detectorinputset.h space.h inputset.cpp inputset.h traininputset.cpp traininputset.h netdef.h dlibnetcfg.h hogsolver.cpp hogsolver.h traineddlibsolver.cpp traineddlibsolver.h trainedcvsolver.cpp trainedcvsolver.h cvnetcfg.h combinedsolver.cpp combinedsolver.h)

add_executable(${PROJECT_NAME} ${SRC} cli.h)
add_executable(CannyTester tester.cpp cli.h)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} dlib::dlib cxxopts)
target_link_libraries(CannyTester PRIVATE ${OpenCV_LIBS})