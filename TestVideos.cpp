//
// Created by ko on 08/02/16.
//

#include "TestVideos.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

#define WIDTH_VIDEO 640
#define HEIGHT_VIDEO 480
#define SKIPPED_PIXELS 50
#define SLIDE_PIXELS 20
#define SLIDE_WINDOW_SIZE 128

void analyseFrames(string dir_to_xml_files, Mat frame, Ptr<SVM> svm);

void TestVideos::testVideo(string dir_to_xml_files, string video) {

    // front back sides bikes
    Ptr<SVM> svm = StatModel::load<SVM>(dir_to_xml_files + "trainedSVM.xml");

    VideoCapture cap;

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
    for(;;)
    {
        if(!cap.grab()) {
            cout << "Done!" << endl;
            break;
        }
        else if (count%15 == 0) {
            cap.retrieve(frame);
            analyseFrames(dir_to_xml_files, frame, svm);
        }
        if (waitKey(30) >= 0) break;
        count++;
    }
}

void analyseFrames(string dir_to_xml_files, Mat frame, Ptr<SVM> svm) {

    Mat grayImg;

    resize(frame, frame, Size(WIDTH_VIDEO,HEIGHT_VIDEO));
    for (int i = SKIPPED_PIXELS; i <= WIDTH_VIDEO - SLIDE_WINDOW_SIZE; i+=SLIDE_PIXELS) {
        bool shouldExit = false;
        for (int j = SKIPPED_PIXELS; j <= HEIGHT_VIDEO - SLIDE_WINDOW_SIZE; j+=SLIDE_PIXELS) {
            Rect myROI(i, j, 128, 128);
            Mat croppedImg = frame(myROI);

            cvtColor(croppedImg, grayImg, CV_RGB2GRAY);
            HOGDescriptor hog(Size(128,128), Size(16,16), Size(8,8), Size(8,8), 9);
            vector<float> descriptors;
            vector<Point> locations;
            hog.compute(grayImg, descriptors, Size(0, 0), Size(0, 0), locations);

            Mat sampleMat = Mat(descriptors);

            int cols = sampleMat.rows;
            int rows = sampleMat.cols;
            Mat newSampleMat(rows, cols, CV_32F);

            Mat tmp = sampleMat.col(0);
            copy(tmp.begin<float>(), tmp.end<float>(), newSampleMat.begin<float>());

            int res = svm->predict(newSampleMat);
            if (res == 1) {

                rectangle(frame, Point(i, j), Point(i+128, j+128), Scalar(32,32,212),1);
                stringstream ss;
                string s;
                // get random file name
                // generated file name is like: /tmp/randomfilename
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

                // write it to relevant directory
                string n = string("/home/ko/Documents/DriveSafe/Data/testing/false-negatives/") + s +
                           string(".jpg");
                imwrite(n, croppedImg);
                shouldExit = true;
            }
            if (shouldExit) break;
        }
        if (shouldExit) break;
    }
    imshow("Video", frame);

}
