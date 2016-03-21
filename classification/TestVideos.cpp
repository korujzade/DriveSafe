//
// Created by ko on 08/02/16.
//

#include "TestVideos.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

// define values
// these values are changed manually to get the most desirable performance
#define WIDTH_VIDEO 256
#define HEIGHT_VIDEO 256
#define SKIPPED_PIXELS 20
#define SLIDE_PIXELS 10
#define SLIDE_WINDOW_SIZE 128
#define SKIPPED_FRAMES 6
#define HOG_WINDOW_SIZE_WIDTH 128
#define HOG_WINDOW_SIZE_HEIGHT 128
#define HOG_BLOCK_SIZE 8
#define HOG_BLOCK_STRIDE 4
#define HOG_CELL_SIZE 4

// declare function that analyses frames
void analyseFrames(string dir_to_xml_files, string dir_to_false_negatives, Mat frame, Ptr<SVM> svm);

// classify data coming from taken video or from camera
void TestVideos::testVideo(string dir_to_xml_files, string dir_to_false_negatives, string video) {

    // svm model
    Ptr<SVM> svm = StatModel::load<SVM>(dir_to_xml_files + "trainedSVM.xml");

    // capture video
    VideoCapture cap;

    // if user choose camera, then open camera
    // else open video file defined
    if (video.compare("0") == 0) {
        cap.open(0);
    }
    else cap.open(video);
    if (!cap.isOpened()) {
        printf("Video/camera can't open!");
        return;
    }
    namedWindow("Video", CV_WINDOW_NORMAL);

    int count = 0;
    Mat frame;
    // retrieve frames while skipping some frames because of the performance
    for(;;)
    {
        if(!cap.grab()) {
            cout << "Done!" << endl;
            break;
        }
        else if (count%SKIPPED_FRAMES == 0) {
            cap.retrieve(frame);
            // send retrieved frame to classify whether there is a bicycle.
            // function requires a directory of xml files which provides extracted features
            // current frame and current model.
            analyseFrames(dir_to_xml_files, dir_to_false_negatives, frame, svm);
        }
        if (waitKey(30) >= 0) break;
        count++;
    }
}

// analyse video frames whether there is a bicycle on it or not using trained svm model
void analyseFrames(string dir_to_xml_files, string dir_to_false_negatives, Mat frame, Ptr<SVM> svm) {

    // two matrix files for gray image and temporary frame
    Mat grayImg;
    Mat frame2;

    // resize image suitable for analysing
    resize(frame, frame, Size(WIDTH_VIDEO,HEIGHT_VIDEO));
    // true if there is a bicycle
    // used for send a push notification to android device
    bool notify = false;

    // slide over a frame by skipping pixels from each sides.
    // the number of the pixels during sliding are usually less than the number of the pixels skipped on each
    // boundary.
    // all values defined already
    for (int i = SKIPPED_PIXELS; i <= WIDTH_VIDEO - SLIDE_WINDOW_SIZE; i+=SLIDE_PIXELS) {
        bool shouldExit = false;
        for (int j = SKIPPED_PIXELS; j <= HEIGHT_VIDEO - SLIDE_WINDOW_SIZE; j+=SLIDE_PIXELS) {

            // initialise pixel values to crop from current frame and send for analysing
            Rect myROI(i, j, HOG_WINDOW_SIZE_WIDTH, HOG_WINDOW_SIZE_HEIGHT);
            // part of current frame will be sent for analysing
            // we need to clone that part to new matrix, so that we will not edit original frame
            Mat croppedImg = frame(myROI).clone();

            // change cropped image to gray scale image
            cvtColor(croppedImg, grayImg, CV_RGB2GRAY);
            // compute hog values for cropped image from current video or camera frame
            HOGDescriptor hog(Size(HOG_WINDOW_SIZE_HEIGHT,HOG_WINDOW_SIZE_WIDTH), Size(HOG_BLOCK_SIZE,HOG_BLOCK_SIZE),
                              Size(HOG_BLOCK_STRIDE,HOG_BLOCK_STRIDE), Size(HOG_CELL_SIZE,HOG_CELL_SIZE), 9);
            vector<float> descriptors;
            hog.compute(grayImg, descriptors, Size(0, 0), Size(0, 0));

            // copy descriptor values to new matrix for further calculations
            Mat sampleMat = Mat(descriptors);

            // because svm models located rows and cols opposite we need to change them in our matrix as well
            int cols = sampleMat.rows;
            int rows = sampleMat.cols;
            Mat newSampleMat(rows, cols, CV_32F);

            Mat tmp = sampleMat.col(0);
            copy(tmp.begin<float>(), tmp.end<float>(), newSampleMat.begin<float>());
            frame2 = frame.clone();

            // use svm model to predict whether there is a bicycle on cropped image of current frame
            int res = svm->predict(newSampleMat);
            // 1 if it is bicycle
            if (res == 1) {
                // draw red boundary over frame to demonstrate bicycle location
                rectangle(frame2, Point(i, j), Point(i+HOG_WINDOW_SIZE_WIDTH, j+HOG_WINDOW_SIZE_HEIGHT),
                          Scalar(32,32,212),1);

                // get random file name for cropped part of the current frame, so that we can save it and use it for
                // hard negative mining, if it is false detection
                // generated file name is like: /tmp/randomfilename
                // reference: http://stackoverflow.com/questions/34165202/generate-random-filename-c
                stringstream ss;
                string s;
                char fn [L_tmpnam];
                tmpnam (fn);
                // parse char to string
                ss << fn;
                ss >> s;
                // change first slash to space
                size_t found = s.find_first_of("/");
                s[found] = ' ';
                found = s.find_first_of("/", found+1);
                // get name after second slash
                string path = s;
                int c = path.rfind('/');
                s = path.substr(c + 1);

                string n = dir_to_false_negatives + s + string(".jpg");
                imwrite(n, croppedImg);
                notify = true;
                shouldExit = true;
            }
            if (shouldExit) break;
        }
        if (shouldExit) break;
    }

    // show current frame
    imshow("Video", frame2);
    if(waitKey(3) >=0 ) return;
    // OPTIONAL: send notification to android device to demonstrate detect-notify principle.
    if (notify)
        system("/home/ko/Documents/DriveSafe/nma.sh \"DriveSafe\" \"Bicycle Detected\" \"Info\" \"2\"");

}



