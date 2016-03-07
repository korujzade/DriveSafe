//
// Created by ko on 05/03/16.
//

#include <opencv2/opencv.hpp>
#ifndef DRIVESAFE_HOGFORSVMLIGHT_H
#define DRIVESAFE_HOGFORSVMLIGHT_H


using namespace std;

class HOGforSVMLight {
public:
    void training(string posImages, string negImages);
};


#endif //DRIVESAFE_HOGFORSVMLIGHT_H
