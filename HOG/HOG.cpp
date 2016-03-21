//
// Created by ko on 07/02/16.
//

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "HOG.h"

using namespace cv;
using namespace std;

// define values
// these values are changed manually during development and  will be changed in future to find out most suitable values
// for the most desirable performance
#define WIDTH_OF_IMAGE 640;
#define HEIGHT_OF_IMAGE 480;
#define HOG_WINDOW_SIZE_WIDTH 128
#define HOG_WINDOW_SIZE_HEIGHT 128
#define HOG_BLOCK_SIZE 8
#define HOG_BLOCK_STRIDE 4
#define HOG_CELL_SIZE 4


// hog descriptor values
// new object class are created for keeping descriptors. This class could be used to store pixel locations of a bicycle
// if it is necessary
class HOG_descriptors {
public:
        vector<float> descriptors;
};

// declare function names
vector<string> getFileNames(string path);
HOG_descriptors getHOGvalues(Mat img);
void releaseHOG(string hogfile, vector <vector<float> > v_descriptors);

// extract features from all training images
void HOG::generateFeatures(string dir_to_read_bikes, string dir_to_read_bike_annotations,
                          string dir_to_read_front_back_view_bikes, string dir_to_negative_images,
                           string dir_to_xml_files, bool isAnnotation)  {

    // arrays to keep file names in each folder
    cout << "reading file names from directories..." << endl;

    vector<string> pos_files2;
    vector<string> pos_files;
    vector<string> annotation_files;
    vector<string> neg_files;

    // if bicycle images has annotation data, then get these files
    if(!isAnnotation) {
        pos_files2 = getFileNames(dir_to_read_front_back_view_bikes);
    } else {
        pos_files = getFileNames(dir_to_read_bikes);
        annotation_files = getFileNames(dir_to_read_bike_annotations);
    }

    // get negative images from predefined directory
    neg_files = getFileNames(dir_to_negative_images);
    cout << "DONE!" << endl;

    // xml files keeping descriptor values
    string posXML = dir_to_xml_files + "pos.xml";
    string negXML = dir_to_xml_files + "neg.xml";

    // 2d arrays to keep descriptor values for bicycle images
    vector <vector<float> > descriptors_pos;
    // negative images
    vector <vector<float> > descriptors_neg;

    // get bicycles for training from images which have annotation data, and extract features from these images
    if (isAnnotation) {
        // extract descriptor values from annotated positive images
        cout << "extracting features from annotated bike images" << endl;
        for (uint i = 0; i < pos_files.size(); i++) {
            const string path_to_pos_file = dir_to_read_bikes + pos_files[i];
            const string path_to_annotation_file = dir_to_read_bike_annotations + annotation_files[i];

            // read ground-truth form of each image
            Mat gtImg = imread(path_to_annotation_file, 0);

            // initialise an image boundaries
            int maxX = 0;
            int minX = WIDTH_OF_IMAGE;
            int maxY = 0;
            int minY = HEIGHT_OF_IMAGE;

            // find coordinates of a bike in current image from relevant annotation of it
            for (int i = 0; i < gtImg.cols; i++) {
                for (int j = 0; j < gtImg.rows; j++) {
                    if (gtImg.at<uchar>(Point(i, j)) == 0) {
                        if (maxX < i) maxX = i;
                        if (maxY < j) maxY = j;
                        if (minX > i) minX = i;
                        if (minY > j) minY = j;
                    }
                }
            }

            // crop a bicycle from current image using ground truth of it and update it
            Mat img = imread(path_to_pos_file);
            Rect myROI(minX, minY, (maxX - minX), (maxY - minY));
            img = img(myROI);

            // find descriptor values of image
            HOG_descriptors hog_desc_values = getHOGvalues(img);

            // write descriptor values to two dimensional vector
            descriptors_pos.push_back(hog_desc_values.descriptors);
        }
        cout << "DONE!" << endl;
        // release hog values to xml files
        cout << "Writing descriptor values for annotated images to xml file..." << endl;
        releaseHOG(posXML, descriptors_pos);
        cout <<"DONE!" << endl;
    }

    // if bicycle images do not have annotations, extract features from images without requesting ground truth data
    if (!isAnnotation) {
        // extract descriptor values from positive images
        cout << "extracting features from front and back view bike images" << endl;
        for (uint i = 0; i < pos_files2.size(); i++) {
            string path_to_pos_file = dir_to_read_front_back_view_bikes + pos_files2[i];

            // read current bicycle image
            Mat img = imread(path_to_pos_file);

            // find descriptor values of image
            HOG_descriptors hog_desc_values = getHOGvalues(img);

            // write descriptor values to two dimensional vector
            descriptors_pos.push_back(hog_desc_values.descriptors);
        }
        cout << "DONE!" << endl;
        cout << "Writing descriptor values for front back view images to xml file..." << endl;
        releaseHOG(posXML, descriptors_pos);
        cout <<"DONE!" << endl;

    }

    // extract descriptor values for negative images
    cout << "extracting features from negative images" << endl;
    for (uint i = 0; i < neg_files.size(); i++) {
        string path_to_neg_file = dir_to_negative_images + neg_files[i];

        Mat img = imread(path_to_neg_file);

        // find descriptor values of images
        HOG_descriptors hog_desc_values = getHOGvalues(img);

        // write descriptor values to two dimensional array
        descriptors_neg.push_back(hog_desc_values.descriptors);
    }
    cout << "DONE!" << endl;
    cout << "Writing descriptor values for negative images to xml file..." << endl;
    releaseHOG(negXML, descriptors_neg);
    cout <<"DONE!" << endl;
}

// get file names in given directory
// reference: http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
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
    // initialise hog descriptor object and a matrix for gray scale image
    HOG_descriptors hog_desc_values;
    Mat grayImg;

    // resize original image to desired size
    resize(img, img, Size(HOG_WINDOW_SIZE_WIDTH, HOG_WINDOW_SIZE_HEIGHT));
    // change image to gray scale image
    cvtColor(img, grayImg, CV_RGB2GRAY);

    // initialise window, block, cell size for hog function and block stride for normalisation
    HOGDescriptor hog(Size(HOG_WINDOW_SIZE_WIDTH,HOG_WINDOW_SIZE_HEIGHT), Size(HOG_BLOCK_SIZE,HOG_BLOCK_SIZE),
                      Size(HOG_BLOCK_STRIDE,HOG_BLOCK_STRIDE), Size(HOG_CELL_SIZE,HOG_CELL_SIZE), 9);
    vector <float> descriptors;
    // compute histogram of oriented gradients of pixels using initialised values and sizes, and write extracted
    // values to descriptors vector
    hog.compute(grayImg, descriptors, Size(0,0), Size(0,0));

    // reference extracted features to the initialised hog descriptors
    hog_desc_values.descriptors = descriptors;
    return hog_desc_values;
}

// write extracted features values from an object to xml file
// reference: http://study.marearts.com/2014/04/the-example-source-code-of-2d-vector.html
void releaseHOG(string hogfile, vector <vector<float> > v_descriptors) {
    FileStorage hogXML(hogfile, FileStorage::WRITE);

    int row = v_descriptors.size();
    int col = v_descriptors[0].size();

   cout << "columns: " << col << " rows: " << row << endl;
    Mat mat(row, col, CV_32F);

    for(int i=0; i< row; ++i)
        memcpy( &(mat.data[col * i * sizeof(float) ]) ,v_descriptors[i].data(),col*sizeof(float));
    //write xml
    write(hogXML, "Descriptor_of_images",  mat);
    hogXML.release();
}