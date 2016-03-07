//
// Created by ko on 08/02/16.
//

#include "TestSVM.h"
#include <dirent.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

clock_t t1, t2;

class PosNeg {
public:
    int pos_count;
    int neg_count;
};

PosNeg getResults(string dir_name, string dir_to_xml_files);

void TestSVM::testRecords(string dir_to_test_bikes, string dir_to_test_negative_images, string dir_to_xml_files) {

    t1 = clock()/(CLOCKS_PER_SEC/1000);
    PosNeg posImgs;
    PosNeg negImgs;

    posImgs = getResults(dir_to_test_bikes, dir_to_xml_files);
    negImgs = getResults(dir_to_test_negative_images, dir_to_xml_files);

    float accuracy = (float)((posImgs.pos_count + negImgs.neg_count))/(float)((posImgs.pos_count + posImgs.neg_count
                                                                               + negImgs.neg_count + negImgs.pos_count));
    float sensitivity = (float)(posImgs.pos_count)/(float)(posImgs.pos_count + posImgs.neg_count);
    float spesificity = (float)(negImgs.neg_count)/(float)(negImgs.pos_count + negImgs.neg_count);

    cout << "true positive results: " << posImgs.pos_count << endl;
    cout << "false positive results: " << posImgs.neg_count << endl;
    cout << "true negative results: " << negImgs.neg_count << endl;
    cout << "false negative results: " << negImgs.pos_count << endl;
    cout << "accuracy: " << accuracy << endl;
    cout << "sensitivity: " << sensitivity << endl;
    cout << "specificity: " << spesificity << endl;

    t2 = clock()/(CLOCKS_PER_SEC/1000);
    float diff ((float)t2 - (float)t1);
    cout << diff << endl;

    return;
}

PosNeg getResults(string dir_name, string dir_to_xml_files) {

    // mainly front back view bikes
    Ptr<SVM> svm = StatModel::load<SVM>(dir_to_xml_files + "trainedSVM.xml");
    const char* dirName = dir_name.c_str();
    namedWindow("Images", CV_WINDOW_NORMAL);

    DIR *dir;
    struct dirent *ent;

    PosNeg posNeg;
    posNeg.pos_count = 0;
    posNeg.neg_count = 0;

    int  count =0;
    if ((dir = opendir (dirName)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            count++;
            if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."));
            else {
                Mat grayImg;
                char filen[100];
                sprintf(filen, "%s/%s",dirName, ent->d_name);
                Mat img = imread(filen);

                resize(img, img, Size(256,256));

                int res =0;
                bool shouldExit = false;
                for (int i = 20; i <=128; i+=10) {
                    for (int j = 20; j<=128; j+=10) {
                        Rect myROI(i, j, 128, 128);
                        Mat croppedImg = img(myROI);

                        cvtColor(croppedImg, grayImg, CV_RGB2GRAY);
                        HOGDescriptor hog(Size(128,128), Size(16,16), Size(8,8), Size(8,8), 9);
                        vector <float> descriptors;
                        hog.compute(grayImg, descriptors, Size(0,0), Size(0,0));

                        Mat sampleMat = Mat(descriptors);

                        int cols = sampleMat.rows;
                        int rows = sampleMat.cols;
                        Mat newsampleMat(rows, cols, CV_32F);

                        Mat tmp = sampleMat.col(0);
                        copy(tmp.begin<float>(), tmp.end<float>(), newsampleMat.begin<float>());

                        res = svm->predict(newsampleMat);

                        if (res == 1) {
                            posNeg.pos_count++;
                            shouldExit = true;
                            rectangle(img, Point(i, j), Point(i+128, j+128), Scalar(32,32,212),1);
                            break;
                        }
                    }
                    if(shouldExit) break;
                }
                imshow("Images", img);
                if(waitKey(3000) >= 0) break;
                if (res == -1) posNeg.neg_count++;
            }
        }
        closedir (dir);
    } else {
        perror ("");
        return posNeg;
    }
      return posNeg;
};


void testMultiScale() {

    HOGDescriptor hog;
    hog.winSize = Size(128,128);
    vector<float> model_v;
   // get_svm_detector(svm, model_v);
    hog.setSVMDetector(model_v);
    vector<Rect> locations;
    locations.clear();
    vector<Point> p;
    //hog.detect(img, p);

    //cvtColor(img, grayImg, CV_RGB2GRAY);
    //Mat draw = grayImg.clone();
    Scalar reference( 0, 255, 0 );
   // draw_locations(draw, locations, reference);

    //imshow("Images", draw);

}



void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}

void draw_locations( Mat & img, const vector< Rect > & locations, const Scalar & color )
{
    if( !locations.empty() )
    {
        vector< Rect >::const_iterator loc = locations.begin();
        vector< Rect >::const_iterator end = locations.end();
        for( ; loc != end ; ++loc )
        {
            rectangle( img, *loc, color, 2 );
        }
    }
}

