import cv2
import numpy as np


#initial min and max HSV filter values.
#these will be changed using trackbars
int H_MIN = 0
int H_MAX = 256
int S_MIN = 0
int S_MAX = 256
int V_MIN = 0
int V_MAX = 256
#default capture width and height
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
#values for location of lines
CENTER_WIDTH = FRAME_WIDTH/2 + 1
CENTER_HEIGHT = FRAME_HEIGHT/2 + 1
deltaCenterX = 120
deltaCenterY = 100
#max number of objects to be detected in frame
MAX_NUM_OBJECTS = 50
#minimum and maximum object area
MIN_OBJECT_AREA = 20 * 20
MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5
#names that will appear at the top of each window
windowName = "Original Image"
windowName1 = "HSV Image"
windowName2 = "Thresholded Image"
windowName3 = "After Morphological Operations""
trackbarWindowName = "Trackbars"

def main()
	capture = cv2.VideoCapture(0)
	
	while(True):
		ret, frame = capture.read()
		
		cv2.imshow("Original Image", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
	capture.release()
	cv2.destroyAllWindows()
	
main()
