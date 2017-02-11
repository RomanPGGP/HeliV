/*
 * Title: Flipped Image Sample (Cascaron)
 * Class: Vision para Robot
 * Instructor: Dr. Jose Luis Gordillo (http://LabRob.mty.itesm.mx/)
 * Code: Manlio Barajas (manlito@gmail.com)
 * Institution: Tec de Monterrey, Campus Monterrey
 * Date: January 10, 2012
 *
 * Description: This program takes input from a camera (recognizable by
 * OpenCV) and it flips it horizontally.
 * "Basic" version uses frequently the "Cv::Mat::At" method which slows down
 * performance. This program has illustrative purposes, provided the existence
 * of cv::flip method.
 *
 * TODO: Validate when source and destination image are the same
 *
 * This programs uses OpenCV http://opencv.willowgarage.com/wiki/
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);

int main(int argc, char *argv[])
{
	/* First, open camera device */
	VideoCapture camera = VideoCapture(0);
	/* Create images where captured and transformed frames are going to be stored */
	Mat currentImage;
	Mat flippedImage;

	while (true)
	{
		/* Obtain a new frame from camera */
		camera.read(currentImage);

		/* Call custom flipping routine. From OpenCV, you could call flip(currentImage, flippedImage, 1) */
		flipImageBasic(currentImage, flippedImage);

		/* Show images */
		imshow("Original", currentImage);
		imshow("Flipped", flippedImage);

		/* If 'x' is pressed, exit program */
		if (waitKey(3) == 'x')
			break;
	}
}

/*
 * This method flips horizontally the sourceImage into destinationImage. Because it uses 
 * "Mat::at" method, its performance is low (redundant memory access searching for pixels).
 */
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols / 2; ++x)
			for (int i = 0; i < sourceImage.channels(); ++i)
			{
				destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i];
				destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = sourceImage.at<Vec3b>(y, x)[i];
			}
}

