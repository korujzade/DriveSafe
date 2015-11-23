cmake_minimum_required(VERSION 2.8)
project( HOG2 )
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( HOG2 HOG2.cpp )
target_link_libraries( HOG2 ${OpenCV_LIBS} )


