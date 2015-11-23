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


struct pn
{
	int p;
	int n;
};

pn testRecords(char folderName[100]);
// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>("../TrainSVM/trainedSVM.xml");
Ptr<SVM> svm2 = StatModel::load<SVM>("../TrainSVM/trainedSVM2.xml");

int main(int, char**)
{
  t1 = clock()/(CLOCKS_PER_SEC/1000);
  pn pospn;
  pn negpn;

  char folderName[100] = "testbikes/bikes";
  char negfn[100] = "none-testing/";

	pospn = testRecords(folderName);
	negpn = testRecords(negfn);

  float accuracy = (float)((pospn.p + negpn.n))/(float)((pospn.p + pospn.n + negpn.n + negpn.p));
  float sensitivity = (float)(pospn.p)/(float)(pospn.p + pospn.n);
  float spesificity = (float)(negpn.n)/(float)(negpn.p + negpn.n);

  cout << "true positive results: " << pospn.p << endl;
  cout << "false positive results: " << pospn.n << endl;
  cout << "true negative results: " << negpn.n << endl;
  cout << "false negative results: " << negpn.p << endl;
  cout << "accuracy: " << accuracy << endl;
  cout << "sensitivity: " << sensitivity << endl;
  cout << "spesificity: " << spesificity << endl; 
  t2 = clock()/(CLOCKS_PER_SEC/1000);
  float diff ((float)t2 - (float)t1);
  cout << diff << endl;
}

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
				HOGDescriptor hog(Size(64,64), Size(16,16), Size(8,8), Size(8,8), 9);
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
        int res2 = svm2->predict(newsampleMat);

				if (res == 1 || res2 == 1)
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