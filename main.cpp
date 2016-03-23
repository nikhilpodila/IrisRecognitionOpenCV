#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/** @function removePupil*/
void removePupil(Mat src2, Mat src)
{
	// Invert the source image and convert to grayscale
	Mat gray;
	cvtColor(~src2, gray, CV_BGR2GRAY);

	// Convert to binary image by thresholding it
	threshold(gray, gray, 220, 255, THRESH_BINARY);

	// Find all contours
	vector<vector<Point> > contours;
	findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// Fill holes in each contour
	drawContours(gray, contours, -1, CV_RGB(255, 255, 255), -1);

	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		Rect rect = boundingRect(contours[i]);
		int radius = rect.width / 2;

		// If contour is big enough and has round shape
		// Then it is the pupil
		if (area >= 30 &&
			abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
			abs(1 - (area / (CV_PI * pow(radius, 2)))) <= 0.2)
		{
			circle(src, Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255, 0, 0), 2);
			rectangle(src2, rect, CV_RGB(0, 0, 255), CV_FILLED);
		}
	}
}

/** @function main */
int main(int argc, char** argv)
{
	Mat src, src_gray;

	/// Read the image
	src = imread("eye_image.jpg", 1);

	if (!src.data)
	{
		return -1;
	}
	Mat src1;
	src.copyTo(src1);
	removePupil(src1,src);
	
	/// Convert it to gray
	cvtColor(src1, src_gray, CV_BGR2GRAY);

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 145, 60, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 255, 0), 3, 8, 0);
	}

	/// Show your results
	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	imshow("Hough Circle Transform Demo", src);

	waitKey(0);
	return 0;
}