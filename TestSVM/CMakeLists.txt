cmake_minimum_required(VERSION 2.8)
project( TestSVM )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( TestSVM TestSVM.cpp )
target_link_libraries( TestSVM ${OpenCV_LIBS} )


