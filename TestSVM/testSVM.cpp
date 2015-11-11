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

using namespace cv;
using namespace cv::ml;
using namespace std;

// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>("../trainsvm/trainedSVM.xml");

int testRecord(char* fileName, char folderName[100])
{
	Mat grayImg;
	char filen[100];
	sprintf(filen, "%s/%s",folderName, fileName);
	//cout << "name: " << filen << endl;
	Mat img = imread(filen);

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

int main(int, char**)
{

    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;

    char folderName[100] = "testbikes";
    DIR *dir;
    struct dirent *ent;
	if ((dir = opendir (folderName)) != NULL) 
	{
		while ((ent = readdir (dir)) != NULL) 
		{
			if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."));
			else
			{
				int res = testRecord(ent->d_name, folderName);
				if (res == 1)
					tp++;
				if (res == 0)
					fp++;
			}
		}
		closedir (dir);
	} 
	else 
	{
	  perror ("");
	  return EXIT_FAILURE;
	}

    cout << "true positive results: " << tp << endl;
    cout << "false positive results: " << fp << endl;

/////////////////////////// Negative Testing//////////////////////////////

    char negfolderName[100] = "../HOG/training/none";
	if ((dir = opendir (negfolderName)) != NULL) 
	{
		while ((ent = readdir (dir)) != NULL) 
		{
			if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."));
			else
			{
				int res = testRecord(ent->d_name, negfolderName);
				if (res == 0)
					tn++;
				if (res == 1)
					fn++;
			}
		}
		closedir (dir);
	} 
	else 
	{
	  perror ("");
	  return EXIT_FAILURE;
	}	

    cout << "true negative results: " << tn << endl;
    cout << "false negative results: " << fn << endl;	

}
