//
// Created by ko on 08/02/16.
//

/*
 * Train extracted features and create svm model
 */

#include "TrainSVM.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

// class has matrixes which keep descriptor values and it's label (positive or negative image)
class Data {
public:
    Mat descriptorValues;
    Mat labels;
};

// declare functions
Data getMatrixofDescriptorValues(string pos, string neg);
void generateSVMModule(Data td, string dir_to_xml_files);


void TrainSVM::createSVMModule(string posXML, string negXML, string dir_to_xml_files) {

    Data td = getMatrixofDescriptorValues(posXML, negXML);
    generateSVMModule(td, dir_to_xml_files);
}

// get descriptor values of positive and negative images
// reference: http://study.marearts.com/2014_11_23_archive.html
Data getMatrixofDescriptorValues(string pos, string neg) {

    // create Data object keeping train data
    Data td;

    cout << "Reading positive and negative HOG descriptor values from xml files ..." << endl;

    //create xml file to read positive descriptor values
    FileStorage pos_xml;
    pos_xml.open(pos, FileStorage::READ);
    // create matrix and write positive descriptor values from xml to it
    Mat pos_mat;
    pos_xml["Descriptor_of_images"] >> pos_mat;
    int pos_row, pos_col;
    pos_row = pos_mat.rows;
    pos_col = pos_mat.cols;

    //release xml file, as we will not need it anymore
    pos_xml.release();
    cout << "Reading positive values DONE!" << endl;

    // negative xml file to read negative descriptor values
    FileStorage neg_xml;
    neg_xml.open(neg, FileStorage::READ);

    //create matrix and write negative descriptor values from xml to it
    Mat neg_mat;
    neg_xml["Descriptor_of_images"] >> neg_mat;

    int neg_rows, neg_cols;
    neg_rows = neg_mat.rows;
    neg_cols = neg_mat.cols;

    //release xml as we will not need it
    neg_xml.release();
    cout << "Reading negative values DONE!" << endl;

    cout << "Row and columns" << endl;
    cout << "positive row: " << pos_row << " positive column: " << pos_col << " negative rows: " << neg_rows;
    cout << " negative columns: " << neg_cols << endl;

    //Make training data which will be used to feed svm
    cout << "Preparing data suitable for SVM training" << endl;
    //descriptor data set
    Mat descriptors_alltogether(pos_row + neg_rows, pos_col, CV_32FC1 );

    pos_mat.copyTo(descriptors_alltogether(Rect(0, 0, pos_col, pos_row)));
    neg_mat.copyTo(descriptors_alltogether(Rect(0, pos_row, pos_col, neg_rows)));

    cout << "rows: " << descriptors_alltogether.rows << " " << "cols: " << descriptors_alltogether.cols << endl;

    // labels 1 and -1 negative and positive data
    Mat labels(pos_row + neg_rows, 1, CV_32SC1, Scalar(-1.0) );
    labels.rowRange(0, pos_row) = Scalar(1.0 );

    td.descriptorValues = descriptors_alltogether;
    td.labels = labels;

    return td;
}

// train and generate svm model and write it to xml files using descriptor values
void generateSVMModule(Data d, string dir_to_xml_files) {

    //Set svm parameters
    cout << "SVM training..." << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON));
    Ptr<TrainData> td =TrainData::create(d.descriptorValues, ROW_SAMPLE, d.labels);
    svm->trainAuto(td);
    printf("Done!\n");

    string fn = dir_to_xml_files + "trainedSVM.xml";
    svm->save(fn);
}