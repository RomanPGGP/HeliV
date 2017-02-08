#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SDL/SDL.h"
/*
 * A simple 'getting started' interface to the ARDrone, v0.2 
 * author: Tom Krajnik
 * The code is straightforward,
 * check out the CHeli class and main() to see 
 */
#include <stdlib.h>
#include "CHeli.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

bool stop = false;
CRawImage *image;
CHeli *heli;
float pitch, roll, yaw, height;
int hover=0;

// Joystick related
SDL_Joystick* m_joystick;
bool useJoystick;
int joypadRoll, joypadPitch, joypadVerticalSpeed, joypadYaw;
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover;
string ultimo = "init";

int Px;
int Py;
int vR;
int vG;
int vB;

Mat imagenClick;

// Here we will store points *CLICK*
vector<Point> points;

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

//codigo del click en pantalla
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    uchar* destination;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
        Px=x;
        Py=y;
        destination = (uchar*) imagenClick.ptr<uchar>(Py);
        vB=destination[Px * 3];
        vG=destination[Px*3+1];
        vR=destination[Px*3+2];
        points.push_back(Point(x, y));
        break;
        case CV_EVENT_MOUSEMOVE:
        break;
        case CV_EVENT_LBUTTONUP:
        break;
        case CV_EVENT_RBUTTONDOWN:
        //flag=!flag;
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
    bool clicked = false;

    /* First, open camera device */    //TESTING ONLY
    VideoCapture webcam;
    webcam.open(0);

    //establishing connection with the quadcopter
    heli = new CHeli();
    
    //this class holds the image from the drone 
    image = new CRawImage(320,240);
    
    // Initial values for control   
    pitch = roll = yaw = height = 0.0;
    joypadPitch = joypadRoll = joypadYaw = joypadVerticalSpeed = 0.0;

    // Destination OpenCV Mat   
    Mat currentImage = Mat(240, 320, CV_8UC3);

    // TESTING ONLY
    /* Create images where captured and transformed frames are going to be stored */
    Mat currentImageWC;
    Mat flippedImageWC;
    Mat im_gray;
    Mat img_bw;

    // Show it  
    imshow("ParrotCam", currentImage);

    // Initialize joystick
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);
    useJoystick = SDL_NumJoysticks() > 0;

    if (useJoystick)
    {
        SDL_JoystickClose(m_joystick);
        m_joystick = SDL_JoystickOpen(0);
    }

    namedWindow("Click");
    setMouseCallback("Click", mouseCoordinatesExampleCallback);

    /* Create main OpenCV window to attach callbacks */  // TESTING ONLY
    namedWindow("ImageWC");
    setMouseCallback("ImageWC", mouseCoordinatesExampleCallback);

    while (!clicked){
        char c = waitKey(5);

        if (c == 'b'){
            clicked = true;
        }

        //image is captured
        heli->renewImage(image);

        /* Obtain a new frame from camera */ //TESTING ONLY 
        webcam >> currentImageWC;

        rawToMat(currentImage, image);
        imshow("ParrotCam", currentImage);
        imagenClick=currentImage;

        /* Show image */
        imshow("Click", imagenClick);
        imshow("ImageWC", currentImageWC);
    }

    while (stop == false)
    {

        // Clear the console
        printf("\033[2J\033[1;1H");

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

        // prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
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

        /* Call custom flipping routine. From OpenCV, you could call flip(currentImage, flippedImage, 1) */
        flipImageBasic(currentImageWC, flippedImageWC);
        imshow("Flipped", flippedImageWC);

        cvtColor(currentImageWC,im_gray,CV_RGB2GRAY);
        imshow("Gray", im_gray);

        img_bw = im_gray > 128;
        imshow("BIN", img_bw);

        // Copy to OpenCV Mat
        rawToMat(currentImage, image);
        imshow("ParrotCam", currentImage);
        imagenClick=currentImage;
        //imshow("Click", imagenClick);
        //imshow("ImageWC", currentImageWC);

        if (currentImage.data) 
        {
            /* Draw all points */
            for (int i = 0; i < points.size(); ++i) {
                circle(currentImageWC, (Point)points[i], 5, Scalar( 0, 0, 255 ), CV_FILLED);
                circle(currentImage, (Point)points[i], 5, Scalar( 0, 0, 255 ), CV_FILLED);
                /*if((points.size() > 1) &&(i != 0)){ //Condicion para no tomar en cuenta el punto -1, que no existe
                    line(imagenClick, (Point)points[i-1],(Point)points[i],Scalar( 0, 0, 255), 3,4,0); //dibuja las lineas entre puntos
                }*/
            }

            /* Show image */
            imshow("Click", imagenClick);
            imshow("ImageWC", currentImageWC);
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