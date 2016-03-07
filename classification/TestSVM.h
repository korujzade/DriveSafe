//
// Created by ko on 08/02/16.
//

#include <opencv2/opencv.hpp>
#ifndef DRIVESAFE_TESTSVM_H
#define DRIVESAFE_TESTSVM_H


using namespace std;


class TestSVM {
public:
    void testRecords(string dir_to_test_bikes, string dir_to_test_negative_images, string dir_to_xml_files);

};


#endif //DRIVESAFE_TESTSVM_H
