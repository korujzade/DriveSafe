//
// Created by ko on 08/02/16.
//

#include <opencv2/opencv.hpp>
#ifndef DRIVESAFE_TRAINSVM_H
#define DRIVESAFE_TRAINSVM_H

using namespace std;

class TrainSVM {

public:
    void createSVMModule(string posXML, string negXML, string dir_to_xml_files);
};


#endif //DRIVESAFE_TRAINSVM_H
