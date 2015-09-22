#include <sstream>
#include <string>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <time.h>

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
const int CENTER_WIDTH = FRAME_WIDTH/2 + 1;
const int CENTER_HEIGHT = FRAME_HEIGHT/2 + 1;
const int deltaCenterX = 120;
const int deltaCenterY = 100;
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
	//vertical lines
    line(frame, Point(CENTER_WIDTH - deltaCenterX, FRAME_HEIGHT), Point(CENTER_WIDTH - deltaCenterX, 0), Scalar(0, 0, 255), 2);
    line(frame, Point(CENTER_WIDTH + deltaCenterX, FRAME_HEIGHT), Point(CENTER_WIDTH + deltaCenterX, 0), Scalar(0, 0, 255), 2);
	//horizontal lines
    line(frame, Point(FRAME_WIDTH, CENTER_HEIGHT - deltaCenterY), Point(0, CENTER_HEIGHT - deltaCenterY), Scalar(0, 255, 255), 2);
    line(frame, Point(FRAME_WIDTH, CENTER_HEIGHT + deltaCenterY), Point(0, CENTER_HEIGHT + deltaCenterY), Scalar(0, 255, 255), 2);
}

void createTrackbars(){
    //create window for trackbars
    namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf(TrackbarName, "H_MIN", 0);
    sprintf(TrackbarName, "H_MAX", 256);
    sprintf(TrackbarName, "S_MIN", 0);
    sprintf(TrackbarName, "S_MAX", 256);
    sprintf(TrackbarName, "V_MIN", 0);
    sprintf(TrackbarName, "V_MAX", 256);
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH),
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->
    createTrackbar("H_MIN", trackbarWindowName, &H_MIN, 256, on_trackbar);
    createTrackbar("H_MAX", trackbarWindowName, &H_MAX, 256, on_trackbar);
    createTrackbar("S_MIN", trackbarWindowName, &S_MIN, 256, on_trackbar);
    createTrackbar("S_MAX", trackbarWindowName, &S_MAX, 256, on_trackbar);
    createTrackbar("V_MIN", trackbarWindowName, &V_MIN, 256, on_trackbar);
    createTrackbar("V_MAX", trackbarWindowName, &V_MAX, 256, on_trackbar);
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
    
    //morphological opening (remove small objects from the foreground)
	erode(thresh, thresh, erodeElement);
	dilate(thresh, thresh, dilateElement);
    
    //morphological closing (fill small holes in the foreground)
	dilate(thresh, thresh, dilateElement);
    erode(thresh, thresh, erodeElement);
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
	//area of largest object found
	//int area = 0; // I changed this to 0 because an object of area 0 doesn't exist

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

void compareValues(int value, int &min, int &max){
    if(value > max)
        max = value;
    if(value < min)
        min = value;
}

void CompareHSV(Mat &HSV){
    int h_min, s_min, v_min, h_max, s_max, v_max;
    h_min = s_min = v_min = 256;
    h_max = s_max = v_max = 0;
    
    for(int x = 0; x < HSV.rows; ++x)
        for(int y = 0; y < HSV.cols; ++y){
            Vec3b hsv = HSV.at<Vec3b>(x,y);
            compareValues(hsv.val[0], h_min, h_max);
            compareValues(hsv.val[1], s_min, s_max);
            compareValues(hsv.val[2], v_min, v_max);
        }
    
    H_MIN = h_min;
    H_MAX = h_max;
    S_MIN = s_min;
    S_MAX = s_max;
    V_MIN = v_min;
    V_MAX = v_max;
}

void waitForObject(VideoCapture &feed, Rect &detectionRectangle, int seconds){
    Mat image;
    time_t start, end;
    
    for(time(&start), time(&end); difftime (end,start) < seconds; time(&end)){
        if(!feed.read(image)){
            std::cout << "Cannot read from camera in waitForObject\n";
            continue;
        }
        rectangle(image, detectionRectangle.tl(), detectionRectangle.br(), Scalar(0, 255, 0), 2, 8, 0);
        std::ostringstream oss;
        oss << "Put object infront of camera Capturing in " << seconds - difftime(end,start) << " Seconds";
        putText(image, oss.str() , Point(0,50), 2, .6, Scalar(0, 255, 0), 2);
        imshow(windowName, image);
    }
}

void setHSV(VideoCapture &feed){
    Mat image, areaOfInterest, HSV;
    //Rect detectionRectangle(FRAME_WIDTH/3, FRAME_HEIGHT/3, FRAME_WIDTH/3, FRAME_HEIGHT/3);
	int scale = 100;
	Rect detectionRectangle(Point(CENTER_WIDTH - scale, CENTER_HEIGHT - scale), Point(CENTER_WIDTH + scale, CENTER_HEIGHT + scale));
    
    //wait 5 seconds to capture object to be detected
    waitForObject(feed, detectionRectangle, 5);
    
    //we will use this image to set HSV values
    while(!feed.read(image)) std::cout << "Cannot read from camera in setHSV\n";
    std::cout << "Got image, now finding HSV values" << std::endl;
    
    //Creates a rectangle of the area we only want to look at in the image
    areaOfInterest = image(detectionRectangle);
    
    //converts rectangle from RGB to HSV
    cvtColor(areaOfInterest, HSV,CV_BGR2HSV);
    
    CompareHSV(HSV);
}

int main(int argc, char* argv[])
{
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	
	//matrix storage for HSV image
	Mat HSV;
	
	//matrix storage for binary threshold image
	Mat threshold;
	
	//x and y values for the location of the object
	int x = 0, y = 0;
	
	//video capture object to acquire webcam feed
	VideoCapture capture(0);
    if(!capture.isOpened()){
        std::cout << "Err: cannot open camera\n";
        return -1;
    }
    
    //set height and width of capture frame
    capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    
    //get HSV values
    setHSV(capture);
    
    //create slider bars for HSV filtering
    createTrackbars();

	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop
	while (1){
		//store image to matrix
        if(!capture.read(cameraFeed)){
            std::cout << "Cannot read camera in main\n";
            continue;
        }
		
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
        
        //draw center line and threshold lines
        drawLines(cameraFeed);

		//show frames 
		imshow(windowName2, threshold);
		imshow(windowName, cameraFeed);
		//imshow(windowName1, HSV);

		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
		waitKey(30);
	}
	return 0;
}

