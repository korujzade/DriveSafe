#include <opencv2/opencv.hpp>
#include "HOG/HOG.h"
#include "training/TrainSVM.h"
#include "classification/TestSVM.h"
#include "classification/TestVideos.h"
#include "svmlight_hog/HOGforSVMLight.h"

/*
 * Main function allows user to whether extract features or used previously extracted features, train features or use
 * previously trained svm model and test system using images, video or real time with a camera.
 */

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
    string dir_to_false_negatives;
    string video;
    string ans;
    bool isAnnotated;

    cout << "Would you like to extract features from images? (yes/no)" << endl;
    cout << "Note that if answer is \"no\", you should have relevant file to keep features in requested folder" << endl;
    cout << "answer: ";
    cin >> ans;

    while (ans.compare("yes") != 0 && ans.compare("no") != 0) {
        cout << "please answer by writing either yes or no: ";
        cin >> ans;
    }

    if (ans.compare("yes") == 0) {

        cout << "Do you have positive images with annotations (ground truth images) (1) or without annotations(2): " << endl;
        cout << "answer: ";
        cin >> ans;

        while (ans.compare("1") != 0 && ans.compare("2") != 0) {
            cout << "please answer by writing either 1 or 2: ";
            cin >> ans;
        }

        if (ans.compare("1") == 0) {
            cout << "Please enter a directory path for bike images: ";
            cin >> dir_to_read_bikes;
            cout << "Please enter a directory path for annotations: ";
            cin >> dir_to_read_bike_annotations;
            cout << "Please enter a directory path for negative images: ";
            cin >> dir_to_read_negative_images;
            cout << "Please enter a directory path for xml files: ";
            cin >> dir_to_xml_files;
            isAnnotated = true;

            dir_to_read_front_back_view_bikes = "/none";
        } else {
            cout << "Please enter a directory path for bike images: ";
            cin >> dir_to_read_front_back_view_bikes;
            cout << "Please enter a directory path for negative images: ";
            cin >> dir_to_read_negative_images;
            cout << "Please enter a directory path for xml files: ";
            cin >> dir_to_xml_files;
            isAnnotated = false;

            dir_to_read_bike_annotations = "/none";
            dir_to_read_bike_annotations = "/none";

        }

        // extracting features from positive and negative images using Histogram of Oriented Gradients
        // function need to path to bikes, their annotations if there is, negative images and
        // directory path where svm models and extracted feature files are going to be located
        HOG hog;
        hog.generateFeatures(dir_to_read_bikes, dir_to_read_bike_annotations,
                             dir_to_read_front_back_view_bikes, dir_to_read_negative_images, dir_to_xml_files, isAnnotated);
    }

    cout << "Would you like to train descriptor values using Linear SVM? (yes/no)" << endl;
    cout << "Please note that if the answer is \"no\", you should have relevant file to keep classified modules" << endl;
    cout << "answer: ";
    cin >> ans;

    while (ans.compare("yes") != 0 && ans.compare("no") != 0) {
        cout << "please answer by writing either yes or no: ";
        cin >> ans;
    }

    if (ans.compare("yes") == 0) {
        cout << "Please enter a directory path for xml files: ";
        cin >> dir_to_xml_files;

        // file names of features from bikes and negative images
        string posXML = dir_to_xml_files + "pos.xml";
        string negXML = dir_to_xml_files + "neg.xml";

        // train svm using extracted features
        TrainSVM trainSVM;
        trainSVM.createSVMModule(posXML, negXML, dir_to_xml_files);
    }

    cout << "Please choose to evaluate system using either given images(1) or videos(2): ";
    cin >> ans;
    while (ans.compare("1") != 0 && ans.compare("2") != 0) {
        cout << "please answer by writing either 1 or 2: ";
        cin >> ans;
    }

    if (ans.compare("1") == 0) {
        cout << "Please enter a directory path to test bike images: ";
        cin >> dir_to_test_bikes;
        cout << "Please enter a directory path to test negative images: ";
        cin >> dir_to_test_negative_images;
        cout << "Please enter a directory path for xml files: ";
        cin >> dir_to_xml_files;

        // classify system using trained svm model and test images
        TestSVM testSVM;
        testSVM.testRecords(dir_to_test_bikes, dir_to_test_negative_images, dir_to_xml_files);
    }

    if (ans.compare("2") == 0) {

        cout << "Please choose to open a camera(1) or video(2): ";
        cin >> ans;
        while (ans.compare("1") != 0 && ans.compare("2") != 0) {
            cout << "please answer by writing either 1 or 2: ";
            cin >> ans;
        }

        if (ans.compare("1") == 0) {
            cout << "Please enter a directory path for xml files" << endl;
            cout << "answer: ";
            cin >> dir_to_xml_files;
            cout << "Please enter a directory path for false negatives" << endl;
            cout << "answer: ";
            cin >> dir_to_false_negatives;

            // classify system using a camera
            TestVideos testVideo;
            testVideo.testVideo(dir_to_xml_files, dir_to_false_negatives, "0");
        } else {
            cout << "Please enter a directory path for xml files: ";
            cin >> dir_to_xml_files;
            cout << "Please enter a directory path for false negatives: ";
            cin >> dir_to_false_negatives;
            cout << "Please enter a path to a video: ";
            cin >> video;

            // classify system real time
            TestVideos testVideo;
            testVideo.testVideo(dir_to_xml_files, dir_to_false_negatives, video);
        }
    }
    return 0;
}
