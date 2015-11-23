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

using namespace cv;
using namespace cv::ml;
using namespace std;

clock_t t1, t2;


int testRecord(char filen[100]);
// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>("/home/korujzade/Desktop/DriveSafe/backend/TrainSVM/trainedSVM.xml");

int main(int argc, char *argv[])
{
  t1 = clock();

  int result = testRecord(argv[1]);

  cout << result << endl;
 
  t2 = clock();
  // float diff ((float)t2 - (float)t1);
  // cout << diff << endl;
}

int testRecord(char filen[100])
{
	Mat img = imread(filen);
	Mat grayImg;

	resize(img, img, Size(128,128));
	cvtColor(img, grayImg, CV_RGB2GRAY);
	HOGDescriptor hog(Size(128,128), Size(8,8), Size(4,4), Size(4,4), 9);
	vector <float> descriptors;
	vector <Point> locations;
	hog.compute(grayImg, descriptors, Size(0,0), Size(0,0), locations);

	Mat sampleMat = Mat(descriptors);

	int cols = sampleMat.rows;
	int rows = sampleMat.cols;
	Mat newsampleMat(rows, cols, CV_32F);

	Mat tmp = sampleMat.col(0);
	copy(tmp.begin<float>(), tmp.end<float>(), newsampleMat.begin<float>());

	int res = svm->predict(newsampleMat);

	return res;
}