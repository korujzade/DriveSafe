#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include "dirent.h"
#include "time.h"
#include <sstream>
#include <cstdio>
#include <string>

using namespace cv;
using namespace cv::ml;
using namespace std;

clock_t t1, t2;

int testRecord(Mat img);
// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>(
		"/home/korujzade/Desktop/DriveSafe/backend/TrainSVM/trainedSVM.xml");
Ptr<SVM> svm2 = StatModel::load<SVM>(
		"/home/korujzade/Desktop/DriveSafe/backend/TrainSVM/trainedSVM2.xml");

int main(int argc, char *argv[]) {
	t1 = clock();
	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	cout <<"tmpmax: "<<  TMP_MAX << endl;
	Mat edges;
	namedWindow("Camera", CV_WINDOW_NORMAL);
	int count = 0;
	for (;;) {
		count++;
		Mat frame;
		bool success = cap.read(frame);
		if (!success) {
			cout << "Cannot read frame" << endl;
			break;
		}
		int result = testRecord(frame);

		if (result == 1) {
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
			string n = string("/home/korujzade/Desktop/DriveSafe/backend") +
					   string("/TestOneImage/false-positives/") + s +
					   string(".jpg");
			imwrite(n, frame);
			cout << "Alert Alert Alert!!! " << endl;
		}

		//cout << result << endl;	
		imshow("Camera", frame);
		if (waitKey(30) >= 0)
			break;
	}

	t2 = clock();
	// float diff ((float)t2 - (float)t1);
	// cout << diff << endl;
}

int testRecord(Mat img) {
	Mat grayImg;

	resize(img, img, Size(128, 128));
	cvtColor(img, grayImg, CV_RGB2GRAY);
	HOGDescriptor hog(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
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

	if (res == 1 || res2 == 1)
		res = 1;

	return res;
}
