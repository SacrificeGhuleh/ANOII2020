cmake_minimum_required(VERSION 3.14)
project(ANOII2020)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Where to find CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Where to find OpenCV
include(MacroOpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Where to find dlib
find_package(dlib REQUIRED)
find_package(Threads REQUIRED)


# Setup path to other libs
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib/")

# Where to find HALF lib
set(CXXOPTS_LIB_DIR "${LIB_DIR}/cxxopts/")
#set(CXXOPTS_LIB_INCLUDE "${CXXOPTS_LIB_DIR}/include/")
#include_directories(${HALF_LIB_INCLUDE})
add_subdirectory(${CXXOPTS_LIB_DIR})

add_subdirectory(src)
