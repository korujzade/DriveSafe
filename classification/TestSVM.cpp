//
// Created by ko on 08/02/16.
//

/*
 * classify test images using trained svm model
 */

#include "TestSVM.h"
#include <dirent.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

#define HOG_WINDOW_SIZE_WIDTH 128
#define HOG_WINDOW_SIZE_HEIGHT 128
#define HOG_BLOCK_SIZE 8
#define HOG_BLOCK_STRIDE 4
#define HOG_CELL_SIZE 4
#define VIDEO_FRAME_WIDTH 256
#define VIDEO_FRAME_HEIGHT 256
#define SKIPPED_PIXELS 10


// used for calculating running time of a function
clock_t t1, t2;

// class keeps the number of the positive and negative images predicted
class PosNeg {
public:
    int pos_count;
    int neg_count;
};

// declare a function
PosNeg getResults(string dir_name, string dir_to_xml_files);

// test positive and negative images to see whether there is a bicycle on it or not
void TestSVM::testRecords(string dir_to_test_bikes, string dir_to_test_negative_images, string dir_to_xml_files) {

    t1 = clock()/(CLOCKS_PER_SEC/1000);

    // objects keeps the number of images predicted as positive and negative images
    // these values are used for calculating accuracy of system
    PosNeg posImgs;
    PosNeg negImgs;

    // get the number of positive and negative prediction
    posImgs = getResults(dir_to_test_bikes, dir_to_xml_files);
    negImgs = getResults(dir_to_test_negative_images, dir_to_xml_files);

    // calculate accuracy, sensitivity and specificity of the system
    float accuracy = (float)((posImgs.pos_count + negImgs.neg_count))/(float)((posImgs.pos_count + posImgs.neg_count
                                                                               + negImgs.neg_count + negImgs.pos_count));
    float sensitivity = (float)(posImgs.pos_count)/(float)(posImgs.pos_count + posImgs.neg_count);
    float spesificity = (float)(negImgs.neg_count)/(float)(negImgs.pos_count + negImgs.neg_count);

    // print out results
    cout << "true positive results: " << posImgs.pos_count << endl;
    cout << "false positive results: " << posImgs.neg_count << endl;
    cout << "true negative results: " << negImgs.neg_count << endl;
    cout << "false negative results: " << negImgs.pos_count << endl;
    cout << "accuracy: " << accuracy << endl;
    cout << "sensitivity: " << sensitivity << endl;
    cout << "specificity: " << spesificity << endl;

    t2 = clock()/(CLOCKS_PER_SEC/1000);
    float diff ((float)t2 - (float)t1);
    cout << diff << endl;

    return;
}

// predict images
PosNeg getResults(string dir_name, string dir_to_xml_files) {

    // svm model
    Ptr<SVM> svm = StatModel::load<SVM>(dir_to_xml_files + "trainedSVM.xml");
    const char* dirName = dir_name.c_str();

    // demonstrate work
    namedWindow("Images", CV_WINDOW_NORMAL);

    DIR *dir;
    struct dirent *ent;

    PosNeg posNeg;
    posNeg.pos_count = 0;
    posNeg.neg_count = 0;

    int  count =0;

    // get each image from directory and analise it
    // for getting files from a directory, dirent library is used
    // reference: http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
    if ((dir = opendir (dirName)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            count++;
            if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."));
            else {
                Mat grayImg;
                char filen[100];
                sprintf(filen, "%s/%s",dirName, ent->d_name);
                Mat img = imread(filen);
                resize(img, img, Size(VIDEO_FRAME_WIDTH,VIDEO_FRAME_HEIGHT));

                int res =0;
                bool shouldExit = false;
                // slide over current image and predict window each time
                for (int i = SKIPPED_PIXELS; i <=HOG_WINDOW_SIZE_HEIGHT; i+=SKIPPED_PIXELS) {
                    for (int j = SKIPPED_PIXELS; j<=HOG_WINDOW_SIZE_WIDTH; j+=SKIPPED_PIXELS) {
                        Mat img2 = img.clone();
                        // visulise sliding window with green boundary rectangle
                        rectangle(img2, Point(i, j), Point(i+HOG_WINDOW_SIZE_HEIGHT, j+HOG_WINDOW_SIZE_WIDTH),
                                  Scalar(25,223,45),1);

                        // initialise current part of the current image while sliding and write it to new matrix
                        Rect myROI(i, j, HOG_WINDOW_SIZE_HEIGHT, HOG_WINDOW_SIZE_WIDTH);
                        Mat croppedImg = img(myROI).clone();
                        // change cropped image to gray scale image
                        cvtColor(croppedImg, grayImg, CV_RGB2GRAY);
                        // initialise hog paramaters exactly same as svm module trained
                        HOGDescriptor hog(Size(HOG_WINDOW_SIZE_HEIGHT,HOG_WINDOW_SIZE_WIDTH),
                                          Size(HOG_BLOCK_SIZE,HOG_BLOCK_SIZE), Size(HOG_BLOCK_STRIDE,HOG_BLOCK_STRIDE),
                                          Size(HOG_CELL_SIZE,HOG_CELL_SIZE), 9);
                        vector <float> descriptors;
                        // compute descriptor values by extracting features from current cropped part
                        hog.compute(grayImg, descriptors, Size(0,0), Size(0,0));

                        // write descriptor values to new matrix so that it could be used for svm prediction
                        Mat sampleMat = Mat(descriptors);
                        int cols = sampleMat.rows;
                        int rows = sampleMat.cols;
                        Mat newsampleMat(rows, cols, CV_32F);
                        Mat tmp = sampleMat.col(0);
                        copy(tmp.begin<float>(), tmp.end<float>(), newsampleMat.begin<float>());

                        // predict cropped part of current image whether it is bicycle or not
                        res = svm->predict(newsampleMat);

                        // if it is bicycle, then visualise it
                        if (res == 1) {
                            // increase the number of the positive images predicted
                            posNeg.pos_count++;
                            shouldExit = true;
                            // show part of the current image bicycle detected
                            rectangle(img2, Point(i, j), Point(i+128, j+128), Scalar(32,32,212),1);
                            // demonstrate it
                            imshow("Images", img2);
                            if(waitKey(3000) >= 0) break;
                            break;
                        }
                        imshow("Images", img2);
                        if(waitKey(30) >= 0) break;
                    }
                    // if bicycle detected on this image, no need further analising, so that we skip rest parts of
                    // the current image
                    if(shouldExit) break;
                }

                if (res == -1) posNeg.neg_count++;
            }
        }
        closedir (dir);
    } else {
        perror ("");
        return posNeg;
    }
      return posNeg;
};


// I have tried with using multi scale HOG and svm light, but result is as not high as my current system.
// I keep this method for further experiments
void testMultiScale() {

    HOGDescriptor hog;
    hog.winSize = Size(HOG_WINDOW_SIZE_HEIGHT,HOG_WINDOW_SIZE_WIDTH);
    vector<float> model_v;
   // get_svm_detector(svm, model_v);
    hog.setSVMDetector(model_v);
    vector<Rect> locations;
    locations.clear();
    vector<Point> p;
    //hog.detect(img, p);

    //cvtColor(img, grayImg, CV_RGB2GRAY);
    //Mat draw = grayImg.clone();
    Scalar reference( 0, 255, 0 );
   // draw_locations(draw, locations, reference);

    //imshow("Images", draw);

}

// get svm model from file and feed hog detector with this model.
// function was officially implemented by opencv.
// function has not been used as it is part of multi scale detection.
// reference: https://github.com/Itseez/opencv/blob/master/samples/cpp/train_HOG.cpp
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}

// function has not been used as it is part of multi scale detection.
// draw boundary for detected object
// function implemented by officially opencv
// reference: https://github.com/Itseez/opencv/blob/master/samples/cpp/train_HOG.cpp
void draw_locations( Mat & img, const vector< Rect > & locations, const Scalar & color )
{
    if( !locations.empty() )
    {
        vector< Rect >::const_iterator loc = locations.begin();
        vector< Rect >::const_iterator end = locations.end();
        for( ; loc != end ; ++loc )
        {
            rectangle( img, *loc, color, 2 );
        }
    }
}

