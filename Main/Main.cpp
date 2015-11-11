#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
	VideoCapture cap(0);
	if(!cap.isOpened())
		return -1;

	Mat edges;
	namedWindow("Camera", CV_WINDOW_NORMAL);
	for(;;)
	{
		Mat frame;

		bool success = cap.read(frame);
		if(!success)
		{
			cout << "Cannot read frame" << endl;
			break;
		}
		flip(frame, frame, 1);
		imshow("Camera", frame);
		imwrite("images/frame.jpg", frame);
		if(waitKey(30) >= 0) break;
	}

	return 0;
}
