import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import cProfile
import re
from array import *
cProfile.run('re.compile("foo|bar")')
ap = argparse.ArgumentParser()
class count_car:
	ap.add_argument("-v", "--video", help="path of video")
ap.add_argument("-a", "--min-area", type=int, default=6000, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
	camera = cv2.VideoCapture("final.mp4") # Capture the video
	time.sleep(0.25)
else:
	camera = cv2.VideoCapture(args["video"])

fps = camera.get(cv2.CAP_PROP_FPS)
firstFrame = None
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
count=0
myList=[]
b = [0] * 5
b1=[]
a=[0] * 30000#for testing
k=[]
k1=[]
k2=[]
k3=[]
k4=[]
k5=[]
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]
distance  = []
distance1 = []
distance2 = []
distance3 = []
distance4 = []
distance5 = []

speed =  []
speed1 = []
i=0
x1=0
y1=0
cY=0
cY1=0
cX1=0
cX=0
total_no_of_frames=0
final=0
f=open('output.txt','w')
while True:
	(grabbed, frame) = camera.read() # Capture individual frames
	text = "Unoccupied"
	if not grabbed:
		break
	total_no_of_frames=total_no_of_frames+1
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Convert each individual frame to Grayscale
	gray = cv2.GaussianBlur(gray, (21, 21), 0)# Apply Gaussian Filter to eliminate noise
	if firstFrame is None:
		firstFrame = gray
		continue
	frameDelta = cv2.absdiff(firstFrame, gray)# Find difference ROI image from two consecutive frames
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]# Apply Binary Thresholding and then Dilate it to increase the quality
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_,cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)# Find contours
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]: #Eliminate contours who have area less than a certain minimum value ,so as to assure only vehicles are selected and not any other unwanted things
				continue
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,255), 3)# Draw bounding rectangles around the selected contours
		M = cv2.moments(c);
		x1 = w/2;
		y1 = h/2;
		cX = x+x1;
		cY = y+y1
		cY=y+y1; # Find x,y,w and h of the bounding rectangle and it's centroid as well and draw a white circle on the centroid
##		print('cX = ',cX)
##		print('cY = ',cY)get the centroid of the yellow rectangles where the car was detected
		a.append(cY)
		a1.append(cY)
		a2.append(cX)
		a3.append(x)
		a4.append(y)
		a5.append(w)
		a6.append(h)
		cv2.circle(frame,(int(cX),int(cY)),7, (255, 255, 255), -1)#print a white circle where the centroid was detected
	text = "Occupied"
	lower_yellow = np.array([0,255,254])
	higher_yellow = np.array([0,255,255])
	no_of_objects = cv2.inRange(frame,lower_yellow,higher_yellow)
	final_no_of_objects = cv2.bitwise_and(frame,frame,mask = no_of_objects)
	gray_final=cv2.cvtColor(final_no_of_objects,cv2.COLOR_BGR2GRAY)
	thresh_final = cv2.threshold(gray_final, 127, 255, cv2.THRESH_BINARY)
	(_,cnts_final, _) = cv2.findContours(gray_final.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	NumeroContornos = len(cnts_final);#find the no. of final contours found
	
	cv2.putText(frame,str(NumeroContornos), (30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)# Whenever a vehicle is detected, keep displaying the counter on the screenof no of vehicles detected at that particular frame only
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),# Keep displaying the current date and time on the screen
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
	cv2.imshow("Parking-Lot", frame)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break	
a1[0]=0
a2[0]=0
# Select tracking points from the image
for i in range(1,len(a1)):
    distance.append((((a1[i]-a1[i-1])**2)+((a2[i]-a2[i-1])**2))**0.5)# Find corresponding displacements by the formula,displacement = sqrt((x1-x2)**2 + (y1-y2)**2)
    distance1.append((((a3[i]-a3[i-1])**2)+((a4[i]-a4[i-1])**2))**0.5)
    distance2.append((((a3[i]+a5[i]-a3[i-1]-a5[i-1])**2)+((a4[i]-a4[i-1])**2))**0.5)
    distance3.append((((a3[i]-a3[i-1])**2)+((a4[i]+a6[i]-a4[i-1]-a4[i-1])**2))**0.5)
    distance4.append((((a3[i]+a5[i]-a3[i-1]-a5[i-1])**2)+((a4[i]+a6[i]-a4[i-1]-a6[i-1])**2))**0.5)
    speed.append(((((a1[i]-a1[i-1])**2)+((a2[i]-a2[i-1])**2))**0.5)/(1000/fps))
    if ((((a1[i]-a1[i-1])**2)+((a2[i]-a2[i-1])**2))**0.5)/33 < 0.66:
            speed1.append(((((a1[i]-a1[i-1])**2)+((a2[i]-a2[i-1])**2))**0.5)/(1000/fps))

for i in range(0,len(distance1)-1):
        distance5.append((distance1[i]+distance2[i]+distance3[i]+distance4[i]+distance[i])/5.0)# Take average of these distances to get the average distance
for i in range(0,len(distance5)-1):
        speed.append(distance5[i]/(1000/fps))# Get the average speed by dividing the average distance by time between two frames
        # Neglect distances which are greater than 20 because these are erroneous displacement vectors( they can be sudden changes while detecting
        #different vehicles which may lead to errors, so they must be neglected)
        #Corresponding to distance of 20, speed = (20/(1000/30)) = 0.66
        if speed[i] < 0.66:
            speed1.append(speed[i])
avg_speed = np.mean(speed1)#  Calculate mean of this speed
resolution = height*width
camera_pixel_size = 572
distance_from_camera_m = 10
focal_length_mm = 5.9
scaling_factor = 1+(distance_from_camera_m*1000/focal_length_mm)
avg_speed_1 = (avg_speed*resolution)/camera_pixel_size# Convert this speed to speed in image plane in metric units by multiplying by the
#resolution(width*height) in pixels and dividing by the camera pixel size which is 572 mm in general on an average for a general camera
avg_speed_2 = avg_speed_1*scaling_factor#Convert this speed to speed in object plane in m/msec by multiplying with the scaling factor
    #Scaling factor = 1+(d/f)
    #d is the distance from camera and it is selected as 10m on an average and f is the focal length of the camera which is 5.9mm on an average
avg_speed_3 = avg_speed_2*1000/(1000*3600)# Multiply this image by 1000 and divide by (1000*3600) to get the speed in the object plane in km/hr
print('fps = ',fps)
print('avg speed in image plane in pixel units in pixels/msec = ',avg_speed)
print('avg speed in image plane in metric units in m/msec = ',avg_speed_1)
print('avg speed in object plane in metric units in m/msec = ',avg_speed_2)
print('avg speed in object plane in metric units in km/hr = ',avg_speed_3)
cProfile.run('re.compile("foo|bar")')

camera.release()
cv2.destroyAllWindows()
