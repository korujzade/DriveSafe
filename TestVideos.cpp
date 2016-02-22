//
// Created by ko on 08/02/16.
//

#include "TestVideos.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

//clock_t t1, t2;
Mat frame;

void analyseFrames(string dir_to_xml_files);


void TestVideos::testVideo(string dir_to_xml_files, int video) {

//    t1 = clock();

    VideoCapture cap(video);

    if (!cap.isOpened()) {
        printf("Video/camera can't open!");
        return;
    }

    namedWindow("Video", CV_WINDOW_NORMAL);
    int count = 0;
    while (1) {
        count++;
        if (count%300== 0) {
            bool success = cap.read(frame);
            if (!success) {
                cout << "Cannot read frame" << endl;
                break;
            }
            analyseFrames(dir_to_xml_files);
            imshow("Video", frame);
            if (waitKey(1000) >= 0)
                break;
        }
    }

//    t2 = clock();
}

void analyseFrames(string dir_to_xml_files) {

    Ptr<SVM> svm = StatModel::load<SVM>(dir_to_xml_files + "trainedSVM1.xml");
    Ptr<SVM> svm2 = StatModel::load<SVM>(dir_to_xml_files + "trainedSVM2.xml");

    Mat grayImg;

    resize(frame, frame, Size(256,256));
    for (int i = 20; i <= 128; i+=10) {
        for (int j = 20; j <= 128; j+=10) {
            //rectangle(frame, Point(i, j), Point(i+128, j+128), Scalar(98,212,32),1);
            Rect myROI(i, j, 128, 128);
            Mat croppedImg = frame(myROI);

            cvtColor(croppedImg, grayImg, CV_RGB2GRAY);
            HOGDescriptor hog(Size(128, 128), Size(8, 8), Size(4, 4), Size(4, 4), 9);
            vector<float> descriptors;
            vector<Point> locations;
            hog.compute(grayImg, descriptors, Size(0, 0), Size(0, 0), locations);

            Mat sampleMat = Mat(descriptors);

            int cols = sampleMat.rows;
            int rows = sampleMat.cols;
            Mat newSampleMat(rows, cols, CV_32F);

            Mat tmp = sampleMat.col(0);
            copy(tmp.begin<float>(), tmp.end<float>(), newSampleMat.begin<float>());

            //int res = svm->predict(newSampleMat);
            int res2 = svm2->predict(newSampleMat);

            if (res2 == 1) {
                rectangle(frame, Point(i, j), Point(i+128, j+128), Scalar(32,32,212),1);
                i = i+118;
                j = j+118;
//                stringstream ss;
//                string s;
//                // get random file name
//                // generated file name is like: /tmp/randomfilename
//                char fn [L_tmpnam];
//                tmpnam (fn);
//                // parse char to string
//                ss << fn;
//                ss >> s;
//                // change first slash to space
//                size_t found = s.find_first_of("/");
//                s[found] = ' ';
//                found = s.find_first_of("/", found+1);
//                // get name after second slash
//                string path = s;
//                int c = path.rfind('/');
//                s = path.substr(c + 1);
//
//                // write it to relevant directory
//                string n = string("/home/ko/Desktop/DriveSafe/backend") +
//                           string("/TestOneImage/false-positives/") + s +
//                           string(".jpg");
//                imwrite(n, croppedImg);
//                cout << "Alert Alert Alert!!! " << endl;
            }

        }
    }

}
