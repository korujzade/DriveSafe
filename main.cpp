#include <opencv2/opencv.hpp>
#include "HOG.h"
#include "TrainSVM.h"
#include "TestSVM.h"
#include "TestVideos.h"

using namespace cv;
using namespace std;

int main (int, char**) {

    string dir_to_read_bikes;
    string dir_to_read_bike_annotations;
    string dir_to_read_front_back_view_bikes;
    string dir_to_read_negative_images;
    string dir_to_xml_files;
    string dir_to_test_bikes;
    string dir_to_test_negative_images;
    string video;
    string ans;

//    cout << "Would you like to extract features from images? (yes/no)" << endl;
//    cout << "Note that if answer is \"no\", you should have relevant file to keep features in requested folder" << endl;
//    cout << "answer: ";
//    cin >> ans;
//
//    while (ans.compare("yes") != 0 && ans.compare("no") != 0) {
//        cout << "please answer by writing either yes or no: ";
//        cin >> ans;
//    }
//
//    if (ans.compare("yes") == 0) {
//        cout << "Please enter a directory path for bike images: ";
//        cin >> dir_to_read_bikes;
//        cout << "Please enter a directory path for annotations: ";
//        cin >> dir_to_read_bike_annotations;
//        cout << "Please enter a directory path for front back view bike images: ";
//        cin >> dir_to_read_front_back_view_bikes;
//        cout << "Please enter a directory path for negative images: ";
//        cin >> dir_to_read_negative_images;
//        cout << "Please enter a directory path for xml files: ";
//        cin >> dir_to_xml_files;
//
//        // extracting features from positive and negative images using Histogram of Oriented Gradients methods
//        HOG hog;
//        hog.generateFeatures(dir_to_read_bikes, dir_to_read_bike_annotations,
//                             dir_to_read_front_back_view_bikes, dir_to_read_negative_images, dir_to_xml_files);
//    }
//
//
//    cout << "Would you like to classify descriptor values using Linear SVM? (yes/no)" << endl;
//    cout << "Please note that if the answer is \"no\", you should have relevant file to keep classified modules" << endl;
//    cout << "answer: ";
//    cin >> ans;
//
//    while (ans.compare("yes") != 0 && ans.compare("no") != 0) {
//        cout << "please answer by writing either yes or no: ";
//        cin >> ans;
//    }
//
//    if (ans.compare("yes") == 0) {
//        cout << "Please enter a directory path for xml files: ";
//        cin >> dir_to_xml_files;
//
//        string posNo1XML = dir_to_xml_files + "pos1.xml";
//        string posNo2XML = dir_to_xml_files + "pos2.xml";
//        string negXML = dir_to_xml_files + "neg.xml";
//        TrainSVM trainSVM;
//        trainSVM.createSVMModule(posNo1XML, posNo2XML, negXML, dir_to_xml_files);
//    }
//
//
//    cout << "Please choose to evaluate system using either given images(1) or videos(2): ";
//    cin >> ans;
//    while (ans.compare("1") != 0 && ans.compare("2") != 0) {
//        cout << "please answer by writing either 1 or 2: ";
//        cin >> ans;
//    }
//
//    if (ans.compare("1") == 0) {
//        cout << "Please enter a directory path to test bike images: ";
//        cin >> dir_to_test_bikes;
//        cout << "Please enter a directory path to test negative images: ";
//        cin >> dir_to_test_negative_images;
//        cout << "Please enter a directory path for xml files: ";
//        cin >> dir_to_xml_files;
//
//        TestSVM testSVM;
//        testSVM.testRecords(dir_to_test_bikes, dir_to_test_negative_images, dir_to_xml_files);
//    }
//
//    if (ans.compare("2") == 0) {
//
//        cout << "Please choose to open a camera(1) or video(2): ";
//        cin >> ans;
//        while (ans.compare("1") != 0 && ans.compare("2") != 0) {
//            cout << "please answer by writing either 1 or 2: ";
//            cin >> ans;
//        }
//
//        if (ans.compare("1") == 0) {
//            cout << "Please enter a directory path for xml files" << endl;
//            cout << "answer: ";
//            cin >> dir_to_xml_files;
//            TestVideos testVideo;
//            testVideo.testVideo(dir_to_xml_files, "0");
//        } else {
//            cout << "Please enter a directory path for xml files: ";
//            cin >> dir_to_xml_files;
//            cout << "Please enter a path to a video: ";
//            cin >> video;
//            TestVideos testVideo;
//            testVideo.testVideo(dir_to_xml_files, video);
//        }
//    }
//
//    system("/home/ko/Downloads/nma.sh \"DriveSafe\" \"Bicycle Detected\" \"asdf\" \"2\"");
//    TestVideos testVideo;
//    testVideo.testVideo("/home/ko/Documents/DriveSafe/xmlFiles/", "/home/ko/Documents/DriveSafe/Data/testing/videos/10.mp4");

//    TestVideos testVideo;
//    testVideo.testVideo("/home/ko/Documents/DriveSafe/xmlFiles/", "0");
//
//    TestSVM testSVM;
//    testSVM.testRecords("/home/ko/Documents/DriveSafe/Data/testing/bikes2/", "/home/ko/Documents/DriveSafe/Data/testing/none_bikes/", "/home/ko/Documents/DriveSafe/xmlFiles/");

    return 0;
}