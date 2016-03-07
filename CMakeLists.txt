cmake_minimum_required(VERSION 3.3)
project(DriveSafe)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

set ( SVMLightTest_HDRS
        /home/ko/Documents/DriveSafe/svmlight_hog/SvmLightLib.h
        /home/ko/Documents/DriveSafe/svmlight/kernel.h
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_learn.h
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_common.h
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_api.h
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_api_types.h
        /home/ko/Documents/DriveSafe/svmlight/svm_learn.h
        /home/ko/Documents/DriveSafe/svmlight/svm_common.h
        )

set ( SVMLightTest_SRCS
        /home/ko/Documents/DriveSafe/svmlight_hog/SvmLightLib.cpp
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_main.c
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_learn.c
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_common.c
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_classify.c
        /home/ko/Documents/DriveSafe/svmlight/svm_struct_api.c
        /home/ko/Documents/DriveSafe/svmlight/svm_learn.c
        /home/ko/Documents/DriveSafe/svmlight/svm_learn_main.c
        /home/ko/Documents/DriveSafe/svmlight/svm_hideo.c
        /home/ko/Documents/DriveSafe/svmlight/svm_common.c
        /home/ko/Documents/DriveSafe/svmlight/svm_classify.c
        )

set ( SVM_HOG_HDRS
        /home/ko/Documents/DriveSafe/HOG/HOG.h
        /home/ko/Documents/DriveSafe/training/TrainSVM.h
        /home/ko/Documents/DriveSafe/classification/TestSVM.h
        /home/ko/Documents/DriveSafe/classification/TestVideos.h
        )

set ( SVM_HOG_SRCS
        /home/ko/Documents/DriveSafe/HOG/HOG.cpp
        /home/ko/Documents/DriveSafe/training/TrainSVM.cpp
        /home/ko/Documents/DriveSafe/classification/TestSVM.cpp
        /home/ko/Documents/DriveSafe/classification/TestVideos.cpp
        )


set(SOURCE_FILES main.cpp )
add_executable(DriveSafe ${SOURCE_FILES} ${SVMLightTest_SRCS} ${SVM_HOG_SRCS})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(DriveSafe ${OpenCV_LIBS})