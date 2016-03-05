//
// Created by ko on 07/02/16.
//

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "HOG.h"

using namespace cv;
using namespace std;

// standard size of annotated bike images
#define WIDTH_OF_IMAGE 640;
#define HEIGHT_OF_IMAGE 480;

// class for hog descriptor values and locations
class HOG_descriptors {
public:
        vector<float> descriptors;
        vector<Point> locations;
};


// declare function names
vector<string> getFileNames(string path);
HOG_descriptors getHOGvalues(Mat img);
void releaseHOG(string hogfile, vector <vector<float> > v_descriptors, vector <vector<Point> > v_locations);


void HOG::generateFeatures(string dir_to_read_bikes, string dir_to_read_bike_annotations,
                          string dir_to_read_front_back_view_bikes, string dir_to_negative_images,
                           string dir_to_xml_files)  {

    // arrays to keep file names in each folder
    cout << "reading file names from directories..." << endl;
    vector<string> pos_files = getFileNames(dir_to_read_bikes);
    vector<string> pos_front_back_files = getFileNames(dir_to_read_front_back_view_bikes);
    vector<string> annotation_files = getFileNames(dir_to_read_bike_annotations);
    vector<string> neg_files = getFileNames(dir_to_negative_images);
    cout << "DONE!" << endl;

    // xml files keeping descriptor values
    string posXMLNo1 = dir_to_xml_files + "pos1.xml";
    string posXMLNo2 = dir_to_xml_files + "pos2.xml";
    string negXML = dir_to_xml_files + "neg.xml";

    // 2d arrays to keep descriptor values and locations
    // annotated images
    vector <vector<float> > v_descriptors_posNo1;
    vector <vector<Point> > v_locations_posNo1;
    // front and back view images
    vector <vector<float> > v_descriptors_posNo2;
    vector <vector<Point> > v_locations_posNo2;
    // negative images
    vector <vector<float> > v_descriptors_neg;
    vector <vector<Point> > v_locations_neg;

    // extract descriptor values from annotated positive images
    cout << "extracting features from annotated bike images" << endl;
    for(uint i =0; i < pos_files.size(); i++) {
        const string path_to_pos_file = dir_to_read_bikes + pos_files[i];
        const string path_to_annotation_file = dir_to_read_bike_annotations + annotation_files[i];

        // read ground-truth form of each image
        Mat gtImg = imread(path_to_annotation_file, 0);

        int maxX = 0;
        int minX = WIDTH_OF_IMAGE;
        int maxY = 0;
        int minY = HEIGHT_OF_IMAGE;

        // find coordinates of each bike in each image from relevant annotation of it
        for(int i=0; i<gtImg.cols; i++) {
            for(int j=0; j<gtImg.rows; j++) {
                if (gtImg.at<uchar>(Point(i,j)) == 0) {
                    if(maxX < i) maxX = i;
                    if(maxY < j) maxY = j;
                    if(minX > i) minX = i;
                    if(minY > j) minY = j;
                }
            }
        }

        // crop object from the image and create newImage
        Mat img = imread(path_to_pos_file);
        Rect myROI(minX, minY, (maxX - minX), (maxY - minY));
        img = img(myROI);

        // find descriptor values of image
        HOG_descriptors hog_desc_values = getHOGvalues(img);

        // right descriptor values to two dimensional vector
        v_descriptors_posNo1.push_back(hog_desc_values.descriptors);
        v_locations_posNo1.push_back(hog_desc_values.locations);
    }
    cout << "DONE!" << endl;

    // extract descriptor values from front back view positive images
    cout << "extracting features from front and back view bike images" << endl;
    for(uint i =0; i < pos_front_back_files.size(); i++)
    {
        string path_to_pos_file = dir_to_read_front_back_view_bikes + pos_front_back_files[i];

        // crop object from the image and create newImage
        Mat img = imread(path_to_pos_file);

        // find descriptor values of image
        HOG_descriptors hog_desc_values = getHOGvalues(img);

        // right descriptor values to two dimensional vector
        v_descriptors_posNo2.push_back(hog_desc_values.descriptors);
        v_locations_posNo2.push_back(hog_desc_values.locations);
    }
    cout << "DONE!" << endl;

    // extract descriptor values for negative images
    cout << "extracting features from negative images" << endl;
    for (uint i = 0; i < neg_files.size(); i++) {
        string path_to_neg_file = dir_to_negative_images + neg_files[i];

        Mat img = imread(path_to_neg_file);

        // find descriptor values of images
        HOG_descriptors hog_desc_values = getHOGvalues(img);

        // write descriptor values to two dimensional array
        v_descriptors_neg.push_back(hog_desc_values.descriptors);
        v_locations_neg.push_back(hog_desc_values.locations);
    }
    cout << "DONE!" << endl;

    // release hog values to xml files
    cout << "Writing descriptor values for annotated images to xml file..." << endl;
    releaseHOG(posXMLNo1, v_descriptors_posNo1, v_locations_posNo1);
    cout <<"DONE!" << endl;
    cout << "Writing descriptor values for front back view images to xml file..." << endl;
    releaseHOG(posXMLNo2, v_descriptors_posNo2, v_locations_posNo2);
    cout <<"DONE!" << endl;
    cout << "Writing descriptor values for negative images to xml file..." << endl;
    releaseHOG(negXML, v_descriptors_neg, v_locations_neg);
    cout <<"DONE!" << endl;
}


// get file names in given directory
vector<string> getFileNames(string path) {

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
    return result;
}


// get descriptor values for an image
HOG_descriptors getHOGvalues(Mat img)
{
    HOG_descriptors hog_desc_values;
    Mat grayImg;

    resize(img, img, Size(128,128));
    cvtColor(img, grayImg, CV_RGB2GRAY);

    // windows size: 128x128; block size: 8x8; block stride: 4x4; cell size: 4x4; nbits: 9
    HOGDescriptor hog(Size(128,128), Size(16,16), Size(8,8), Size(8,8), 9);
    vector <float> descriptors;
    vector <Point> locations;
    hog.compute(grayImg, descriptors, Size(0,0), Size(0,0));

    vector<float> detector;
    hog.setSVMDetector(detector);

    hog_desc_values.descriptors = descriptors;
    hog_desc_values.locations = locations;

    return hog_desc_values;
}

// extract to xml file
void releaseHOG(string hogfile, vector <vector<float> > v_descriptors, vector <vector<Point> > v_locations) {
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