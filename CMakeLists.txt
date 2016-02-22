cmake_minimum_required(VERSION 3.3)
project(DriveSafe)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp HOG.cpp HOG.h HOG.cpp HOG.h TrainSVM.cpp TrainSVM.h TestSVM.cpp TestSVM.h TestVideos.cpp TestVideos.h)
add_executable(DriveSafe ${SOURCE_FILES} HOG.cpp HOG.h HOG.cpp HOG.h TrainSVM.cpp TrainSVM.h TestSVM.cpp TestSVM.h TestVideos.cpp TestVideos.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(DriveSafe ${OpenCV_LIBS})