#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "dirent.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

clock_t t1, t2;

struct pn
{
    int p;
    int n;
};

pn testRecords(char folderName[50]);
// load trained svm xml
Ptr<SVM> svm = StatModel::load<SVM>("/home/ko/Desktop/DriveSafe/backend/TrainSVM/trainedSVM.xml");
Ptr<SVM> svm2 = StatModel::load<SVM>("/home/ko/Desktop/DriveSafe/backend/TrainSVM/trainedSVM2.xml");

int main(int, char**)
{
    t1 = clock()/(CLOCKS_PER_SEC/1000);
    pn pospn;
    pn negpn;

    char folderName[60];
    String tmp = "testbikes/front-back-bikes";
    strncpy(folderName, tmp.c_str(), sizeof(folderName));
    folderName[sizeof(folderName) - 1] = 0;

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

pn testRecords(char folderName[59])
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

                resize(img, img, Size(256,256));
                int res1 =0;
                int res2 =0;
                bool shouldExit = false;
                for (int i = 0; i <=128; i+=10) {
                    for (int j = 0; j<=128; j+=10) {
                        // rectangle(img, Point(i, j), Point(i+10, j+10), Scalar(98,212,32),1);
                        Rect myROI(i, j, 128, 128);
                        Mat croppedImg = img(myROI);

                        cvtColor(croppedImg, grayImg, CV_RGB2GRAY);
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

                        res1 = svm->predict(newsampleMat);
                        res2 = svm2->predict(newsampleMat);

                        if (res2 == 1)
                        {
                            newpn.p++;
                            shouldExit = true;
                            rectangle(img, Point(i, j), Point(i+128, j+128), Scalar(32,32,212),1);
                            break;
                        }
                    }
                    if(shouldExit) break;
                }

   				// imshow("test", img);
   				// if(waitKey(3000) >= 0) break;

                //cout << "Done!" << endl;
                if (res2 == -1)
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