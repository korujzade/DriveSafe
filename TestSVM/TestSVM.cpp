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

struct pn
{
	int p;
	int n;
};

// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>("../TrainSVM/trainedSVM.xml");

pn testRecords(char folderName[100])
{
	DIR *dir;
  struct dirent *ent;

	pn newpn;
	newpn.p = 0;
	newpn.n = 0;

	if ((dir = opendir (folderName)) != NULL) 
	{
		while ((ent = readdir (dir)) != NULL) 
		{
			if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."));
			else
			{
				Mat grayImg;
				char filen[100];
				sprintf(filen, "%s/%s",folderName, ent->d_name);
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

				if (res == 1)
					newpn.p++;
				if (res == 0)
					newpn.n++;
			}
		}
		closedir (dir);
	} 
	else 
	{
	  perror ("");
	  return newpn;
	}
	return newpn;
}

int main(int, char**)
{
  pn pospn;
  pn negpn;

  char folderName[100] = "testbikes";
  char negfn[100] = "../HOG/training/none/";

	pospn = testRecords(folderName);
	negpn = testRecords(negfn);

  float accuracy = (float)((pospn.p + negpn.n))/(float)((pospn.p + pospn.n + negpn.n + negpn.p));

  cout << "true positive results: " << pospn.p << endl;
  cout << "false positive results: " << pospn.n << endl;
  cout << "true negative results: " << negpn.n << endl;
  cout << "false negative results: " << negpn.p << endl;

  
  cout << "accuracy: " << accuracy << endl;
}
