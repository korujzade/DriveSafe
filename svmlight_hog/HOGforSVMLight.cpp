//
// Created by ko on 05/03/16.
//

#include <opencv2/opencv.hpp>
#include "HOGforSVMLight.h"
#include "SvmLightLib.h"

using namespace cv;
using namespace std;

vector<string> getFileNames(string path);


void HOGforSVMLight::training(string posImages, string negImages) {
    // we are going to use HOG to obtain feature vectors:

    vector<string> pos_files = getFileNames(posImages);
    vector<string> neg_files = getFileNames(negImages);


    HOGDescriptor hog;
    hog.winSize = Size(32,48);

    // and feed SVM with them:
    SVMLight::SVMTrainer svm("/home/ko/Documents/DriveSafe/features.dat");

    size_t posCount = 0, negCount = 0;
    for (size_t i = 0; i < pos_files.size(); ++i)
    {

        const string path_to_pos_file = posImages + pos_files[i];

        Mat img = imread(path_to_pos_file,CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data)
            break;

        // obtain feature vector:
        vector<float> featureVector;
        hog.compute(img, featureVector, Size(8, 8), Size(0, 0));

        // write feature vector to file that will be used for training:
        svm.writeFeatureVectorToFile(featureVector, true);                  // true = positive sample
        posCount++;

        // clean up:
        featureVector.clear();
        img.release();              // we don't need the original image anymore
    }

    for (size_t i = 0; i < neg_files.size(); ++i)
    {
        const string path_to_neg_file = negImages + neg_files[i];

        Mat img = imread(path_to_neg_file,CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data)
            break;

        // obtain feature vector:
        vector<float> featureVector;
        hog.compute(img, featureVector, Size(8, 8), Size(0, 0));

        svm.writeFeatureVectorToFile(featureVector, false);
        negCount++;
        // clean up:
        img.release();              // we don't need the original image anymore

    }

    std::cout   << "finished writing features: "
    << posCount << " positive and "
    << negCount << " negative samples used";
    std::string modelName("/home/ko/Documents/DriveSafe/classifier.dat");
    svm.trainAndSaveModel(modelName);
    std::cout   << "SVM saved to " << modelName;
}

