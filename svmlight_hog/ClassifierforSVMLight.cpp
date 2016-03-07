//
// Created by ko on 05/03/16.
//

#include "ClassifierforSVMLight.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void ClassifierforSVMLight::classify(string model) {

    HOGDescriptor hog;
    hog.winSize = Size(32,48);
    SVMLight::SVMClassifier c(model);
    vector<float> descriptorVector = c.getDescriptorVector();
    hog.setSVMDetector(descriptorVector);

}

