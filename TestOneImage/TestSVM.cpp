#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

clock_t t1, t2;
Mat frame;

void testRecord();
// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>(
        "/home/ko/Desktop/DriveSafe/backend/TrainSVM/trainedSVM.xml");
Ptr<SVM> svm2 = StatModel::load<SVM>(
        "/home/ko/Desktop/DriveSafe/backend/TrainSVM/trainedSVM2.xml");

int main(int argc, char *argv[]) {
    t1 = clock();
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    Mat edges;
    namedWindow("Camera", CV_WINDOW_NORMAL);
    int count = 0;
    for (;;) {
        count++;

        bool success = cap.read(frame);
        if (!success) {
            cout << "Cannot read frame" << endl;
            break;
        }



        testRecord();
    }

    t2 = clock();
}

void testRecord() {
    Mat grayImg;

    resize(frame, frame, Size(512,512));
    for (int i = 0; i < 512; i+=128) {
        for (int j = 0; j < 512; j+=128 ) {
            rectangle(frame, Point(i, j), Point(i+128, j+128), Scalar(98,212,32),1);
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
            Mat newsampleMat(rows, cols, CV_32F);

            Mat tmp = sampleMat.col(0);
            copy(tmp.begin<float>(), tmp.end<float>(), newsampleMat.begin<float>());

            int res = svm->predict(newsampleMat);
            int res2 = svm2->predict(newsampleMat);


            if (res2 == 1) {
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
                string n = string("/home/ko/Desktop/DriveSafe/backend") +
                           string("/TestOneImage/false-positives/") + s +
                           string(".jpg");
                imwrite(n, croppedImg);
                cout << "Alert Alert Alert!!! " << endl;

            }

            imshow("Camera", frame);
              if (waitKey(30) >= 0)
                  break;
        }
    }
}
