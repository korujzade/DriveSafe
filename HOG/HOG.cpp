// This files takes positives and negatives images from respective folders and create feature descriptor values
// of them and save them to pos.xml and neg.xml files



#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include <vector>

using namespace cv;
using namespace std;

int main(int, char**)
{

	// file names to take from folder
	char fileName[100] = "bike_";
	char currentGTFileName[100];
	char currentRealFileName[100];

	// descriptor values file
	char posXML[100] = "pos.xml";
	// the number of images
    int imgNum = 300;

	vector <vector<float> > v_descriptors;
	vector <vector<Point> > v_locations;

	for(int i = 227; i <= imgNum; i++)
	{
		if (i <= 9)
		{
			sprintf(currentGTFileName, "training/annotations/%s00%d_gt.jpg", fileName, i);
			sprintf(currentRealFileName, "training/bike/%s00%d.bmp", fileName, i);
		}	
		else if (i <= 99)
		{
			sprintf(currentGTFileName, "training/annotations/%s0%d_gt.jpg", fileName, i);
			sprintf(currentRealFileName, "training/bike/%s0%d.bmp", fileName, i);
		}	
		else
		{
			sprintf(currentGTFileName, "training/annotations/%s%d_gt.jpg", fileName, i);
			sprintf(currentRealFileName, "training/bike/%s%d.bmp", fileName, i);
		}	
	
		// read ground truth form of a image	
		Mat gtImg = imread(currentGTFileName, 0);

		int maxX = 0;
		int minX = 640;
		int maxY = 0;
		int minY = 480;

		// find coordinates of the object in a image
		for(int i=0; i<gtImg.cols; i++)
		{	
		    for(int j=0; j<gtImg.rows; j++)
			{	
				if (gtImg.at<uchar>(Point(i,j)) == 0)
				{
					if(maxX < i)
						maxX = i;
					if(maxY < j)
						maxY = j;
					if(minX > i)
						minX = i;
					if(minY > j)
						minY = j;
				}		
			}	
		}
		// crop object from the image and create newImage
		Mat img = imread(currentRealFileName);
		Rect myROI(minX, minY, (maxX - minX), (maxY - minY));
		img = img(myROI);

		Mat grayImg;

		resize(img, img, Size(128,128));

		cvtColor(img, grayImg, CV_RGB2GRAY);

		HOGDescriptor hog(Size(128,128), Size(8,8), Size(4,4), Size(4,4), 9);
		vector <float> descriptors;
		vector <Point> locations;
		hog.compute(grayImg, descriptors, Size(0,0), Size(0,0), locations);

		v_descriptors.push_back(descriptors);
		v_locations.push_back(locations);

		imshow("test", img);

		waitKey(50000);
	}

	FileStorage hogXML(posXML, FileStorage::WRITE);

	int row = v_descriptors.size(), col = v_descriptors[0].size();

	printf("col=%d, row=%d\n", col, row );
	Mat M(row, col, CV_32F);

	 for(int i=0; i< row; ++i)    
	   memcpy( &(M.data[col * i * sizeof(float) ]) ,v_descriptors[i].data(),col*sizeof(float));  
	 //write xml  
	 write(hogXML, "Descriptor_of_images",  M);  

	 hogXML.release(); 



//////////////////////////////NEGATIVES/////////////////////////////

	char negFileName[100] = "training/none/bg_graz_";
	char currentNegFileName[100];

	char negXML[100] = "neg.xml";

	int negImgNum = 380;

	vector <vector<float> > v_descriptors_neg;
	vector <vector<Point> > v_locations_neg; 

	for (int i = 1; i <= negImgNum; i++)
	{
		if (i <= 9)
			sprintf(currentNegFileName, "%s00%d.bmp", negFileName, i);
		else if (i <= 99)
			sprintf(currentNegFileName, "%s0%d.bmp", negFileName, i);
		else
			sprintf(currentNegFileName, "%s%d.bmp", negFileName, i);


		for (int j = 0; j <= 256; j = j + 128)
		{
			for (int k = 0; k <= 256; k = k + 128)
			{
				Mat img = imread(currentNegFileName);
				Rect myROI(j, k, 128, 128);
				img = img(myROI);

				Mat grayImg;

				resize(img, img, Size(128,128));

				cvtColor(img, grayImg, CV_RGB2GRAY);

				HOGDescriptor hog(Size(128,128), Size(8,8), Size(4,4), Size(4,4), 9);
				vector <float> descriptors;
				vector <Point> locations;
				hog.compute(grayImg, descriptors, Size(0,0), Size(0,0), locations);

				v_descriptors_neg.push_back(descriptors);
				v_locations_neg.push_back(locations);

				imshow("neg", img);

				waitKey(1);

			}				
		}
	}

	FileStorage hogXML_neg(negXML, FileStorage::WRITE);

	int row_neg = v_descriptors_neg.size(), col_neg = v_descriptors_neg[0].size();

	printf("col=%d, row=%d\n", col_neg, row_neg );
	Mat M_neg(row_neg, col_neg, CV_32F);

	 for(int i=0; i< row_neg; ++i)    
	   memcpy( &(M_neg.data[col_neg * i * sizeof(float) ]) ,v_descriptors_neg[i].data(),col_neg*sizeof(float));  
	 //write xml  
	 write(hogXML_neg, "Descriptor_of_images",  M_neg);  

	 hogXML_neg.release(); 
}	
