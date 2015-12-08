#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include <vector>
#include <dirent.h>
#include <iostream>

using namespace cv;
using namespace std;

struct HOG_descriptors
{
	vector<float> descriptors;
	vector<Point> locations;
};

vector<string> files(string path);
HOG_descriptors hog(Mat img);
void releaseHOG(string hogfile, vector <vector<float> > v_descriptors, vector <vector<Point> > v_locations);

int main(int, char**)
{
	// path to folders
	string pos_path= "/home/korujzade/Desktop/DriveSafe/backend/HOG/training/bikes/";
	string annotation_path = "/home/korujzade/Desktop/DriveSafe/backend/HOG/training/annotations/";
	string neg_path = "/home/korujzade/Desktop/DriveSafe/backend/HOG/training/none/";

	// arrays for files in each folder
	vector<string> pos_files = files(pos_path);
	vector<string> annotation_files = files(annotation_path);
	vector<string> neg_files = files(neg_path);

	// xml files to keep descriptor values
	string posXML = "pos2.xml";
	string negXML = "neg2.xml";

	// arrays to keep descriptor values and locations
	vector <vector<float> > v_descriptors;
	vector <vector<Point> > v_locations;
	vector <vector<float> > v_descriptors_neg;
	vector <vector<Point> > v_locations_neg;

	// extract descriptor values from positive imagess
	for(uint i =0; i < pos_files.size(); i++)
	{
		string path_to_pos_file = pos_path + pos_files[i];
		string path_to_annotation_file = annotation_path + annotation_files[i];

		// read ground-truth form of each image
		Mat gtImg = imread(path_to_annotation_file, 0);

		int maxX = 0;
		int minX = 640;
		int maxY = 0;
		int minY = 480;

		// find coordinates of the object in a image from an annotation
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
		Mat img = imread(path_to_pos_file);
		Rect myROI(minX, minY, (maxX - minX), (maxY - minY));
		img = img(myROI);

		// find descriptor values of image
		HOG_descriptors newhog = hog(img);

		// right descriptor values to two dimensional vector
		v_descriptors.push_back(newhog.descriptors);
		v_locations.push_back(newhog.locations);
	}

	// extract descriptor values for negative images
	for (uint i = 0; i < neg_files.size(); i++)
	{
		string path_to_neg_file = neg_path + neg_files[i];
		for (int j = 0; j < 256; j = j + 256)
		{	
			for (int k = 0; k <= 256; k = k + 256)
			{
				Mat img = imread(path_to_neg_file);
				Rect myROI(j, k, 128, 128);
				img = img(myROI);
				
				// find descriptor values of images
				HOG_descriptors newhog = hog(img);

				// write descriptor values to two dimensional array
				v_descriptors_neg.push_back(newhog.descriptors);
				v_locations_neg.push_back(newhog.locations);
			}
		}		
	}

	// release hog values to xml files
	releaseHOG(posXML, v_descriptors, v_locations);
	releaseHOG(negXML, v_descriptors_neg, v_locations_neg);
}

// find files from a folder and add alphabetically to array
vector<string> files(string path)
{
	DIR *dir;
	struct dirent *ent;
	vector <string> result;

	dir = opendir(path.empty()? "." : path.c_str());
	if (dir)
	{
		while ((ent = readdir (dir)) != NULL)
			if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."));
			else result.push_back(string(ent->d_name));
		sort(result.begin(), result.end());
	}
	// cout << result.front() << endl;
	// cout << result.back() << endl;
	return result;
}

// create descriptor values for an image
HOG_descriptors hog(Mat img)
{
	HOG_descriptors newHOG;
	Mat grayImg;

	resize(img, img, Size(128,128));
	cvtColor(img, grayImg, CV_RGB2GRAY);

	// windows size: 128x128; block size: 8x8; block stride: 4x4; cell size: 4x4; nbits: 9
	HOGDescriptor hog(Size(64,64), Size(16,16), Size(8,8), Size(8,8), 9);
	vector <float> descriptors;
	vector <Point> locations;
	hog.compute(grayImg, descriptors, Size(0,0), Size(0,0), locations);

	newHOG.descriptors = descriptors;
	newHOG.locations = locations;

	return newHOG;
}

// extract to xml file
void releaseHOG(string hogfile, vector <vector<float> > v_descriptors, vector <vector<Point> > v_locations)
{
	FileStorage hogXML(hogfile, FileStorage::WRITE);

	int row = v_descriptors.size(), col = v_descriptors[0].size();

	printf("col=%d, row=%d\n", col, row );
	Mat M(row, col, CV_32F);

	 for(int i=0; i< row; ++i)
	   memcpy( &(M.data[col * i * sizeof(float) ]) ,v_descriptors[i].data(),col*sizeof(float));
	 //write xml
	 write(hogXML, "Descriptor_of_images",  M);

	 hogXML.release();
}