#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int, char**)
{
 
 //Read Hog feature from XML file
 printf("1. Feature data xml load\n");

 //create xml to read
 FileStorage read_PositiveXml;
 read_PositiveXml.open("../HOG/pos.xml", FileStorage::READ);
 FileStorage read_NegativeXml;
 //Positive Mat
 Mat pMat;
 read_PositiveXml["Descriptor_of_images"] >> pMat;
 //Read Row, Cols
 int pRow,pCol;
 pRow = pMat.rows; pCol = pMat.cols;

 read_NegativeXml.open("../HOG/neg.xml", FileStorage::READ);

  //Negative Mat
 Mat nMat;
 read_NegativeXml["Descriptor_of_images"] >> nMat;
 //Read Row, Cols
 int nRow,nCol;
 nRow = nMat.rows; nCol = nMat.cols;

 //Rows, Cols printf
 printf("   pRow=%d pCol=%d, nRow=%d nCol=%d\n", pRow, pCol, nRow, nCol );
 //release
 read_PositiveXml.release();
 //release
 read_NegativeXml.release();

 //Make training data for SVM
 printf("Make training data for SVM\n");
 //descriptor data set
 Mat posneg_descriptors_mat( pRow + nRow, pCol, CV_32FC1 ); //in here pCol and nCol is descriptor number, so two value must be same;

 pMat.copyTo(posneg_descriptors_mat(Rect(0,0, pCol, pRow)));
 nMat.copyTo(posneg_descriptors_mat(Rect(0,pRow, pCol, nRow)));

 cout << "rows: " << posneg_descriptors_mat.rows << " " << "cols: " << posneg_descriptors_mat.cols << endl;

 //data labeling
 Mat labels( pRow + nRow, 1, CV_32SC1, Scalar(-1.0) );
      labels.rowRange( 0, pRow ) = Scalar( 1.0 );

 //Set svm parameter
 printf("SVM training\n");
 Ptr<SVM> svm = SVM::create();
 svm->setType(SVM::C_SVC);
 svm->setKernel(SVM::LINEAR);
 svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
 Ptr<TrainData> td =TrainData::create(posneg_descriptors_mat, ROW_SAMPLE, labels);
 svm->trainAuto(td);

 svm->save("trainedSVM.xml");

 //Trained data save
 printf("5. SVM xml save\n");
 // /svm.save( "trainedSVM.xml" );
}
