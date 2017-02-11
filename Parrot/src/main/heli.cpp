#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SDL/SDL.h"
#include <stdlib.h>
#include "CHeli.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

// ---------------------------------------------------- Joystick related
SDL_Joystick* m_joystick;
bool useJoystick;
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover;
int joypadRoll = 0; 
int joypadPitch = 0; 
int joypadVerticalSpeed = 0; 
int joypadYaw = 0;
// ----------------------------------------------------

// ---------------------------------------------------- Matrices to store the image
Mat clickedImage;  
Mat currentImage = Mat(240, 320, CV_8UC3);
Mat flippedImage;
Mat grayImage;
Mat binaryImage;
Mat YIQImage;
Mat HSVImage;

// -- TESTING ONLY
Mat currentImageWC;
Mat flippedImageWC;
Mat grayImageWC;
Mat binaryImageWC;
Mat HSVImageWC;
Mat YIQImageWC;
// ----------------------------------------------------

// ---------------------------------------------------- Binary Related
int const max_value = 255;
int const max_BINARY_value = 255;
int threshold_value = 128;

char* binaryWindowName = "Binary";
char* trackbar_type = "Type: \n 0: Binary";
char* trackbar_value = "Value";
// ----------------------------------------------------

CRawImage *image;
CHeli *heli;
vector<Point> points;               //-- Here we will store points *CLICK*
bool stop = false;                  //-- Stops program
bool clicked = false;               //-- Freezes stream
float pitch = 0.0;
float roll = 0.0;
float yaw = 0.0;
float height = 0.0;
int hover = 0;

int Px;
int Py;
int vR;
int vG;
int vB;

void Threshold_Demo( int, void* )
{
    int threshold_type = 0; //Binary
    //threshold(grayImage, binaryImage, threshold_value, max_BINARY_value,threshold_type );
    //imshow(binaryWindowName, binaryImage);

    //-- T*
    threshold(grayImageWC, binaryImageWC, threshold_value, max_BINARY_value,threshold_type);
    imshow(binaryWindowName, binaryImageWC);
}

//Convert RGB to YIQ (Ju)
void convert2YIQ(const Mat &sourceImage, Mat &destinationImage)
{
    int Y;
    int I;
    int Q;
    int R;
    int G;
    int B;

    if (destinationImage.empty())
        destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

    for (int y = 0; y < sourceImage.rows; ++y)      
        for (int x = 0; x < sourceImage.cols; ++x)  
        {
            B = sourceImage.at<Vec3b>(y, x)[1];
            G = sourceImage.at<Vec3b>(y, x)[2];
            R = sourceImage.at<Vec3b>(y, x)[3];
            
            //Conversion from RGB to YIQ
            Y = 0.299*R + 0.587*G + 0.114*B;
            I = 0.596*R - 0.275*G - 0.321*B;
            Q = 0.212*R - 0.523*G + 0.311*B;
            
            //Changing the channel value to the YIQ world
            destinationImage.at<Vec3b>(y, x)[1] = Y;
            destinationImage.at<Vec3b>(y, x)[2] = I;
            destinationImage.at<Vec3b>(y, x)[3] = Q;
        }
}

//Histogram 
int histogram(const Mat &sourceImage)
{
    Mat src, dst;
	
    src = sourceImage;

	if( !src.data )
    { 
        return -1; 
    }

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; 
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;
    //Mat hist;

    // Draw the histograms for B, G and R
    int hist_w = 512; 
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    Mat histImager( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImageg( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImageb( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImageb.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImageg.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImager.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
      line( histImageb, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                   Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                   Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImageg, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                   Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                   Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImager, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                   Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                   Scalar( 0, 0, 255), 2, 8, 0  );
    }

    /// Display
    namedWindow("calcHist R", CV_WINDOW_AUTOSIZE );
    imshow("calcHist R", histImager );

    namedWindow("calcHist G", CV_WINDOW_AUTOSIZE );
    imshow("calcHist G", histImageg );

    namedWindow("calcHist B", CV_WINDOW_AUTOSIZE );
    imshow("calcHist B", histImageb );

    return 0;
}


// Convert CRawImage to Mat
void rawToMat( Mat &destImage, CRawImage* sourceImage)
{   
    uchar *pointerImage = destImage.ptr(0);
    
    for (int i = 0; i < 240*320; i++)
    {
        pointerImage[3*i] = sourceImage->data[3*i+2];
        pointerImage[3*i+1] = sourceImage->data[3*i+1];
        pointerImage[3*i+2] = sourceImage->data[3*i];
    }
}

void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    uchar* destination;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            Px=x;
            Py=y;
            destination = (uchar*) clickedImage.ptr<uchar>(Py);
            vB=destination[Px * 3];
            vG=destination[Px*3+1];
            vR=destination[Px*3+2];
            points.push_back(Point(x, y));
        break;
        case CV_EVENT_RBUTTONDOWN:
            clicked = true;
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

int main(int argc,char* argv[])
{
    VideoCapture webcam;                            //-- T*
    webcam.open(0);                                 //-- T*

    heli = new CHeli();                             //-- Establishing connection with the quadcopter 
    image = new CRawImage(320,240);                 //-- Holds the image from the drone
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);   //-- Initialize joystick
    useJoystick = SDL_NumJoysticks() > 0;

    if (useJoystick)
    {
        SDL_JoystickClose(m_joystick);
        m_joystick = SDL_JoystickOpen(0);
    }
  
    imshow("ParrotCam", currentImage);

    //-- Create main OpenCV window to attach callbacks
    namedWindow("Click");
    setMouseCallback("Click", mouseCoordinatesExampleCallback);
    namedWindow("ImageWC");                                         //-- T*
    setMouseCallback("ImageWC", mouseCoordinatesExampleCallback);   //-- T*

    //-- While user doesn't click in image, keep streaming
    while (!clicked){
        waitKey(5);

        heli->renewImage(image);
        webcam >> currentImageWC;                   //-- T*
        
        rawToMat(currentImage, image);
        imshow("ParrotCam", currentImage);

        clickedImage = currentImage;
        
        imshow("Click", clickedImage);
        imshow("ImageWC", currentImageWC);

    }

    while (stop == false)
    {
        printf("\033[2J\033[1;1H");                 //-- Clear the console

        cout << "useJOystick" << endl;

        if (useJoystick)
        {
            SDL_Event event;
            SDL_PollEvent(&event);

            joypadRoll = SDL_JoystickGetAxis(m_joystick, 2);
            joypadPitch = SDL_JoystickGetAxis(m_joystick, 3);
            joypadVerticalSpeed = SDL_JoystickGetAxis(m_joystick, 1);
            joypadYaw = SDL_JoystickGetAxis(m_joystick, 0);
            joypadTakeOff = SDL_JoystickGetButton(m_joystick, 1);
            joypadLand = SDL_JoystickGetButton(m_joystick, 2);
            joypadHover = SDL_JoystickGetButton(m_joystick, 0);
        }

        //-- Prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
        printf("===================== Parrot Basic Example =====================\n\n");
        fprintf(stdout, "Angles  : %.2lf %.2lf %.2lf \n", helidata.phi, helidata.psi, helidata.theta);
        fprintf(stdout, "Speeds  : %.2lf %.2lf %.2lf \n", helidata.vx, helidata.vy, helidata.vz);
        fprintf(stdout, "Battery : %.0lf \n", helidata.battery);
        fprintf(stdout, "Hover   : %d \n", hover);
        fprintf(stdout, "Joypad  : %d \n", useJoystick ? 1 : 0);
        fprintf(stdout, "  Roll    : %d \n", joypadRoll);
        fprintf(stdout, "  Pitch   : %d \n", joypadPitch);
        fprintf(stdout, "  Yaw     : %d \n", joypadYaw);
        fprintf(stdout, "  V.S.    : %d \n", joypadVerticalSpeed);
        fprintf(stdout, "  TakeOff : %d \n", joypadTakeOff);
        fprintf(stdout, "  Land    : %d \n", joypadLand);
        fprintf(stdout, "Navigating with Joystick: %d \n", navigatedWithJoystick ? 1 : 0);
        cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor RGB: ("<<vR<<","<<vG<<","<<vB<<")"<<endl;

        //-- Flip image
        flipImageBasic(currentImageWC, flippedImageWC);     //-- T*
        imshow("Flipped", flippedImageWC);                  //-- T*
        //flipImageBasic(currentImage, flippedImage);       
        //imshow("Flipped", flippedImage);                  

        //-- Gray image
        cvtColor(currentImageWC,grayImageWC,CV_RGB2GRAY);   //-- T*
        imshow("Gray", grayImageWC);                        //-- T*
        //cvtColor(currentImage,grayImage,CV_RGB2GRAY);     
        //imshow("Gray", grayImage);                        

        //-- Binary image
        binaryImageWC = grayImageWC > 128;                  //-- T*
        imshow("BIN", binaryImageWC);                       //-- T*
        //binaryImage = grayImage > 128;                      
        //imshow("BIN", binaryImage);                       

        //-- Binary image alternative
        namedWindow(binaryWindowName, CV_WINDOW_AUTOSIZE);  //-- Create window
        //-- Create Trackbar to choose type of Threshold

        createTrackbar( trackbar_value,
                        binaryWindowName, &threshold_value,
                        max_value, Threshold_Demo );

        Threshold_Demo(0, 0);                               //-- Call the function to initialize

        //-- HSV
        cvtColor(currentImageWC, HSVImageWC, CV_RGB2HSV);   //-- T*
        imshow("HSV", HSVImageWC);                          //-- T*
        //cvtColor(currentImage, HSVImage, CV_RGB2HSV);
        //imshow("HSV", HSVImage);

        //-- YIQ
        convert2YIQ(currentImageWC, YIQImageWC);            //-- T*
        imshow("YIQ", YIQImageWC);                          //-- T*
        //convert2YIQ(currentImage, YIQImage);
        //imshow("YIQ", YIQImage);

    	//Histogram
    	/*if(histogram(currentImage)==-1)
    		cout << "No image data.. " <<endl;
    	else
    	   histogram(currentImage);*/

                                                            //-- T*
        if(histogram(currentImageWC)==-1)
            cout << "No image data.. " <<endl;
        else
            histogram(currentImageWC);

        // Copy to OpenCV Mat
        rawToMat(currentImage, image);
        imshow("ParrotCam", currentImage);
        clickedImage = currentImage;
        
        if (currentImageWC.data) 
        {
            /* Draw all points */
            for (int i = 0; i < points.size(); ++i) {
                circle(currentImageWC, (Point)points[i], 5, Scalar( 0, 0, 255 ), CV_FILLED);        //--T*
               // circle(currentImage, (Point)points[i], 5, Scalar( 0, 0, 255 ), CV_FILLED);
                if((points.size() > 1) &&(i != 0)){ //Condicion para no tomar en cuenta el punto -1, que no existe
                    //line(clickedImage, (Point)points[i-1],(Point)points[i],Scalar( 0, 0, 255), 3,4,0); //dibuja las lineas entre puntos
                    line(currentImageWC, (Point)points[i-1],(Point)points[i],Scalar( 0, 0, 255), 3,4,0); 
                }
            }

            /* Show image */
            //imshow("Click", clickedImage);
            imshow("ImageWC", currentImageWC);              //--T*
        }
        else
        {
            cout << "No image data.. " << endl;
        }


        char key = waitKey(5);

        switch (key) 
        {
            case 'a': yaw = -20000.0; break;
            case 'd': yaw = 20000.0; break;
            case 'w': height = -20000.0; break;
            case 's': height = 20000.0; break;
            case 'q': heli->takeoff(); break;
            case 'e': heli->land(); break;
            case 'z': heli->switchCamera(0); break;
            case 'x': heli->switchCamera(1); break;
            case 'c': heli->switchCamera(2); break;
            case 'v': heli->switchCamera(3); break;
            case 'j': roll = -20000.0; break;
            case 'l': roll = 20000.0; break;
            case 'i': pitch = -20000.0; break;
            case 'k': pitch = 20000.0; break;
            case 'h': hover = (hover + 1) % 2; break;
            case 27: stop = true; break;
            default: pitch = roll = yaw = height = 0.0;
        }

        if (joypadTakeOff) 
        {
            heli->takeoff();
        }
        
        if (joypadLand) 
        {
            heli->land();
        }
            //hover = joypadHover ? 1 : 0;

            //setting the drone angles
        if (joypadRoll != 0 || joypadPitch != 0 || joypadVerticalSpeed != 0 || joypadYaw != 0)
        {
            heli->setAngles(joypadPitch, joypadRoll, joypadYaw, joypadVerticalSpeed, hover);
            navigatedWithJoystick = true;
        }
        else
        {
            heli->setAngles(pitch, roll, yaw, height, hover);
            navigatedWithJoystick = false;
        }

        usleep(15000);
    }

    heli->land();
    SDL_JoystickClose(m_joystick);
    delete heli;
    delete image;
    return 0;
}

