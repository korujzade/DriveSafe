#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int, char**)
{
 
 //Read Hog feature from XML file
cout << "Reading positive and negative HOG descriptor values from xml files ..." << endl;

 //create xml to read
 FileStorage read_PositiveXml;
 read_PositiveXml.open("../HOG/pos.xml", FileStorage::READ);
 //Positive Mat
 Mat pMat;
 read_PositiveXml["Descriptor_of_images"] >> pMat;
 //Read Row, Cols
 int pRow,pCol;
 pRow = pMat.rows; pCol = pMat.cols;

 //release
 read_PositiveXml.release();
 cout << "Reading positive values DONE!" << endl;

 FileStorage read_NegativeXml;
 read_NegativeXml.open("../HOG/neg.xml", FileStorage::READ);
 
  //Negative Mat
 Mat nMat;
 read_NegativeXml["Descriptor_of_images"] >> nMat;
 
 //Read Row, Cols
 int nRow,nCol;
 nRow = nMat.rows; nCol = nMat.cols;

//release
 read_NegativeXml.release();
 cout << "Reading negative values DONE!" << endl;

 //Rows, Cols printf
 cout << "Row and columns" << endl;
 printf("   pRow=%d pCol=%d, nRow=%d nCol=%d\n", pRow, pCol, nRow, nCol );
 
 //Make training data for SVM
 printf("Making training data for SVM ...\n");
 //descriptor data set
 Mat posneg_descriptors_mat( pRow + nRow, pCol, CV_32FC1 ); //in here pCol and nCol is descriptor number, so two value must be same;

 pMat.copyTo(posneg_descriptors_mat(Rect(0,0, pCol, pRow)));
 nMat.copyTo(posneg_descriptors_mat(Rect(0,pRow, pCol, nRow)));

 cout << "rows: " << posneg_descriptors_mat.rows << " " << "cols: " << posneg_descriptors_mat.cols << endl;

 //data labeling
 Mat labels( pRow + nRow, 1, CV_32SC1, Scalar(-1.0) );
      labels.rowRange( 0, pRow ) = Scalar( 1.0 );

 //Set svm parameter
 printf("SVM training ...\n");
 Ptr<SVM> svm = SVM::create();
 svm->setType(SVM::C_SVC);
 svm->setKernel(SVM::LINEAR);
 svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON));
 Ptr<TrainData> td =TrainData::create(posneg_descriptors_mat, ROW_SAMPLE, labels);
 svm->trainAuto(td);
 printf("Done!\n");

 svm->save("trainedSVM.xml");
}
