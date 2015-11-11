cmake_minimum_required(VERSION 2.8)
project( testSVM )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( testSVM testSVM.cpp )
target_link_libraries( testSVM ${OpenCV_LIBS} )


