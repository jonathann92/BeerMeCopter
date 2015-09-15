#include <sstream>
#include <string>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <pthread.h>
#include <unistd.h>
#define NUM_THREADS 5

using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//values for location of lines
const int CENTER = FRAME_WIDTH/2 + 1;
const int deltaCenter = 120;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

pthread_mutex_t cmutex, fmutex;

//Matrix to store each frame of the webcam feed
Mat cameraFeed_G;
//matrix storage for HSV image
Mat HSV_G;
//matrix storage for binary threshold image
Mat threshold_G;


void on_trackbar(int, void*)
{//This function gets called whenever a
    // trackbar position is changed
}

string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void drawLines(Mat &frame) {
    line(frame, Point(CENTER, FRAME_HEIGHT), Point(CENTER, 0), Scalar(255, 0, 0), 2);
    line(frame, Point(CENTER - deltaCenter, FRAME_HEIGHT), Point(CENTER - deltaCenter, 0), Scalar(0, 0, 255), 2);
    line(frame, Point(CENTER + deltaCenter, FRAME_HEIGHT), Point(CENTER + deltaCenter, 0), Scalar(0, 0, 255), 2);
}

void createTrackbars(){
    //create window for trackbars
    namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf(TrackbarName, "H_MIN", H_MIN);
    sprintf(TrackbarName, "H_MAX", H_MAX);
    sprintf(TrackbarName, "S_MIN", S_MIN);
    sprintf(TrackbarName, "S_MAX", S_MAX);
    sprintf(TrackbarName, "V_MIN", V_MIN);
    sprintf(TrackbarName, "V_MAX", V_MAX);
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH),
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->
    createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
    createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
    createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
    createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
    createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
    createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);
}

void drawObject(int x, int y, Mat &frame){
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)
    circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
    if (y - 25>0)
        line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
    else
        line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
    if (y + 25<FRAME_HEIGHT)
        line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
    else
        line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
    if (x - 25>0)
        line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
    else
        line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
    if (x + 25<FRAME_WIDTH)
        line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
    else
        line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);
    
    putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
}

void morphOps(Mat &thresh){
    
    //create structuring element that will be used to "dilate" and "erode" image.
    //erodeElement is of size 12x12 to get rid of superfluous noise and make object tracking cleaner
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(12, 12));
    
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
    
    erode(thresh, thresh, erodeElement);
    erode(thresh, thresh, erodeElement);
    
    dilate(thresh, thresh, dilateElement);
    dilate(thresh, thresh, dilateElement);
}


/**
 *	Purpose: to track the object inside the camera feed and draw a rectangle around the entire object
 *
 *	Return:  returns the area of the largest object that is being tracked for depth calculation
 **/

double objectFound(int &x, int &y, vector< vector<Point> > &contours, vector<Vec4i> &hierarchy, int &largestObject){
    double refArea = 0;
    largestObject = 0;
    
    for(int i = 0; i >=0; i = hierarchy[i][0]){
        Moments moment = moments ((cv::Mat)contours[i]);
        double area = moment.m00;
        
        //if the area is less than 20 px by 20px then it is probably just noise
        //if the area is the same as the 3/2 of the image size, probably just a bad filter
        //we only want the object with the largest area so we safe a reference area each
        //iteration and compare it to the area in the next iteration.
        if(area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA && area > refArea) {
            x = moment.m10 / area;
            y = moment.m01 / area;
            refArea = area;
            largestObject = i;
        } else
            refArea = 0;
    }
    
    return refArea;
}

void drawRectangle(int &x, int &y, Mat &cameraFeed, vector< vector<Point> > &contours, int &largestObject ){
    //aproximate contours to the polygon
    vector<Point> contours_poly;
    Rect boundRect;
    approxPolyDP(Mat(contours[largestObject]), contours_poly, 3, true);
    boundRect = boundingRect(Mat(contours_poly));
    //draws rectangle around largest object
    rectangle(cameraFeed, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0);
    
    
    
    /* Draws rectangle around all detected objects
     //approximate contours to the polygon
     vector<vector<Point>> contours_poly(contours.size());
     vector<Rect> boundRect (contours.size());
     for(int i = 0; i < contours.size(); ++i) {
     approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
     boundRect[i] = boundingRect(Mat(contours_poly[i]));
     rectangle(cameraFeed, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
     }
     */
}

int findFilteredObjects(int &x, int &y, vector< vector<Point> > &contours, vector<Vec4i> &hierarchy, Mat &cameraFeed) {
    int largestObject = -1;
    double area;
    if(area = objectFound(x, y, contours, hierarchy, largestObject)) {
        char* trackingString = "Tracking Object";
        putText(cameraFeed, trackingString, Point(0,50), 2, 1, Scalar(0, 255, 0), 2);
        std::cout << "area of largest object: " << area << std::endl;
        
        //draw object location on screen
        drawObject(x, y, cameraFeed);
        
        //calculate the area of the largest tracked object
        drawRectangle(x, y, cameraFeed, contours, largestObject);
    }
    else {
        putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
    }
    
    return area;
}

int trackFilteredObject(int &x, int &y, Mat &threshold, Mat &cameraFeed){
    Mat temp;
    threshold.copyTo(temp);
    
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    
    //use moments method to find our filtered object
    int size = hierarchy.size();
    if (size && size < MAX_NUM_OBJECTS)
        return findFilteredObjects(x, y, contours, hierarchy, cameraFeed);
    
    return 0;
}

void error(string msg, int err){
    std::cout << "Error, return code from " << msg << ": " << err << std::endl;
    exit(-1);
}

void setGlobalFrames(Mat cameraFeed, Mat HSV, Mat threshold){
    //pthread_mutex_lock(&fmutex);
    cameraFeed_G = cameraFeed;
    HSV_G = HSV;
    threshold_G = threshold;
    //pthread_mutex_unlock(&fmutex);
}

void *TrackObject(void *args){
    VideoCapture *capture;
    capture = (VideoCapture*)args;
    
    //Matrix to store each frame of the webcam feed
    Mat cameraFeed;
    
    //matrix storage for HSV image
    Mat HSV;
    
    //matrix storage for binary threshold image
    Mat threshold;
    
    //x and y values for the location of the object
    int x = 0, y = 0;
    
    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop
    
    while (1){
        //store image to matrix
        pthread_mutex_lock(&cmutex);
        capture->read(cameraFeed);
        pthread_mutex_unlock(&cmutex);
        
        //convert frame from BGR to HSV colorspace
        cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
        
        //filter HSV image between values and store filtered image to
        //threshold matrix
        inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
        
        //perform morphological operations on thresholded image to eliminate noise
        //and emphasize the filtered object(s)
        morphOps(threshold);
        
        //pass in thresholded frame to our object tracking function
        //this function will return the x and y coordinates of the
        //filtered object
        int trackedObjectArea = trackFilteredObject(x, y, threshold, cameraFeed);
        
        setGlobalFrames(cameraFeed, HSV, threshold);
        
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
    //create slider bars for HSV filtering
    createTrackbars();
    
    //video capture object to acquire webcam feed
    //VideoCapture *capture;
    VideoCapture capture;
    
    capture.open(0);
    
    //open capture object at location zero (default location for webcam)
    //capture->open(0);
    
    //set height and width of capture frame
    capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    
    pthread_t threadid[NUM_THREADS];
    pthread_mutex_init(&cmutex, NULL);
    pthread_mutex_init(&fmutex, NULL);
    
    for(int i = 0; i < 5; ++i){
        int rc = pthread_create(&threadid[i], NULL, TrackObject, (void *)&capture);
        if(rc)
            error("pthread_create", rc);
    }
    
    sleep(2); // sleep so threads can fill global frames
    while(1){
        imshow(windowName2, threshold_G);
        drawLines(cameraFeed_G);
        imshow(windowName, cameraFeed_G);
        imshow(windowName1, HSV_G);
        
        waitKey(30);
    }
    
    for(int i =0; i < NUM_THREADS; ++i){
        int rc = pthread_join(threadid[i], NULL);
        if(rc)
            error("pthread_join", rc);
    }
    
    return 0;
}

