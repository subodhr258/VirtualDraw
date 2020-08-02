import numpy as np 
import cv2
import imutils
import os 
import pkg_resources.py2_warn
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
spectrum = cv2.imread("Spectrum.png")
if spectrum is None:
	print("'Spectrum.png' image not found.")
	exit(1)
spectrum = cv2.resize(spectrum,None,fx = 0.8,fy = 0.8)
(spectrum_x,spectrum_y,_)= spectrum.shape
helpme = False
thickness = 5
pts = []
pts.append([0,0,0]) 
pts.append(thickness)
allpts = []
allptsinput = True
savecount = 0
pixel = (0,0,0)
MYDIR = "Drawings"
CHECK_FOLDER = os.path.isdir(MYDIR)

def nothing(x):
	pass

def isolateColor(img,hsvLower,hsvUpper):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_red = np.array([hsvLower,120,50])
	upper_red = np.array([hsvUpper,255,255])
	mask1 = cv2.inRange(hsv,lower_red,upper_red)

	mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
	mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE, np.ones((2,2),np.uint8))
	mask2 = cv2.bitwise_not(mask1)

	# res1 = cv2.bitwise_and(gray,gray,mask = mask1)
	#finalOutput = cv2.addWeighted(res1,1,res2,1,0)
	gray2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
	res1 = cv2.bitwise_and(img,img,mask = mask1)
	res2 = cv2.bitwise_and(gray2,gray2,mask = mask2)
	final = cv2.addWeighted(res1,1,res2,1,0)
	return final
# If folder doesn't exist, then create it.

if not CHECK_FOLDER:
	os.makedirs(MYDIR)

while os.path.isfile(r"Drawings\drawing%i.jpg"%savecount):
	savecount+=1
# if a video path was not supplied, grab the reference
# to the webcam
vs = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
cv2.namedWindow("Artwork")
cv2.createTrackbar("Upper","Artwork",0,180,nothing)
cv2.createTrackbar("Lower","Artwork",0,180,nothing)

while vs.isOpened():
	_,frame = vs.read()
	frame = cv2.flip(frame,1)
	if frame is None:
		break
	frame = imutils.resize(frame, width=600)
	x,y,_ = frame.shape
	x = x//2
	y = y//2
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	cv2.rectangle(frame,(x+20,y+20),(x-20,y-20),(0,255,255),2)
	#cv2.circle(frame, (x,y), 5, (255, 0, 255), -1)
	hueLower = cv2.getTrackbarPos('Lower','Artwork')
	hueUpper = cv2.getTrackbarPos('Upper','Artwork')
	frame = isolateColor(frame,hueLower,hueUpper)
	cv2.putText(frame,"Spectrum:",(10,20),fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,fontScale = 0.7,color=(255,255,255),thickness=1)
	cv2.putText(frame,"Press M to manually input color",(20,90),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
	cv2.putText(frame,"Press C to capture object color",(20,60),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
	cv2.putText(frame,"Press Q to quit",(20,120),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
	frame[0:0+spectrum_x,130:130+spectrum_y] = spectrum
	cv2.imshow("Artwork",frame)
	#cv2.imshow("blurred",blurred)
	#cv2.imshow("hsv",hsv)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("c"):
		for i in range(10):
			for j in range(10):
				pixel += hsv[x+i,y+j] + hsv[x-i,y-j] + hsv[x+i,y-j] + hsv[x-i,y+j]
		pixel = pixel//400
		#print("hsv:",pixel)
		#print("bgr:",frame[x,y])
		#print("bgr blurred:",blurred[x,y])
		if pixel[0] < 5:
			pixel[0] = 5
		colorLower = tuple([pixel[0]-5, 86, 6])
		colorUpper = tuple([pixel[0]+5, 255, 255])
		#print(colorLower)
		#print(colorUpper)
		break
	elif key == ord("q") or key==27:#27 means escape button
		vs.release()
		break
	elif key == ord("m"):
		colorLower = (hueLower,86,6)
		colorUpper = (hueUpper,255,255)
		break
cv2.destroyAllWindows()
#allow the camera or video file to warm up



################################################Drawing#######################################################
################################################Starts########################################################
################################################Here##########################################################
##############################################################################################################


cv2.namedWindow("Frame")
cv2.createTrackbar("R","Frame",0,255,nothing)
cv2.createTrackbar("G","Frame",0,255,nothing)
cv2.createTrackbar("B","Frame",0,255,nothing)
# keep looping
while vs.isOpened():
	# grab the current frame
	_,frame = vs.read()
	frame = cv2.flip(frame,1)
	# handle the frame from VideoCapture or VideoStream
	
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	paint = np.zeros_like(frame) + 20
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	# print("colorupper:",colorUpper)
	# print("colorlower:",colorLower)
	
	mask = cv2.inRange(hsv, np.float32(colorLower), np.float32(colorUpper))
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	b = cv2.getTrackbarPos('B','Frame')
	g = cv2.getTrackbarPos('G','Frame')
	r = cv2.getTrackbarPos('R','Frame')
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		if M["m00"]!=0:
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		pts[0]=(b,g,r)
		pts[1]=thickness
		# only proceed if the radius meets a minimum size
		if radius > 30:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, center, int(radius),(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (255, 0, 255), -1)
			# update the points queue
			pts.append(center)
			allptsinput=True

		else:
			if(len(pts)>0):
				if pts[(len(pts)-1)] is not None:
					pts.append(None)
					allptsinput = False
			pts=[(b,g,r),thickness]
		if pts[len(pts)-1] is not None and allptsinput == True and pts not in allpts:
			allpts.append(pts)
			
			
		# loop over the set of tracked points
	for point in allpts:
		for i in range(3,len(point)):

			# if either of the tracked points are None, ignore
			# them
			if point[i - 1] is None or point[i] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			#thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)

			cv2.line(frame, point[i - 1],point[i],point[0], point[1])
			cv2.line(paint, point[i - 1],point[i],point[0], point[1])
		# show the frame to our screen


	cv2.putText(frame,"Press h to toggle help",(20,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color=(255,255,255),thickness=2)
	cv2.putText(frame,"Color:",(400,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color=(255,255,255),thickness=1)
	cv2.putText(frame,"Thickness:",(400,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color=(255,255,255),thickness=1)
	cv2.line(frame,(565,55),(595,55),(b,g,r),thickness)
	if helpme == True:
		cv2.putText(frame,"Press: q to quit",(20,60),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		cv2.putText(frame,"r to refresh",(20,100),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		cv2.putText(frame,"[ to decrease thickness",(20,140),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		cv2.putText(frame,"] to increase thickness",(20,180),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		cv2.putText(frame,"u to undo",(20,220),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		cv2.putText(frame,"s to save with background",(20,260),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		cv2.putText(frame,"p to save without background",(20,300),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale = 1,color=(255,255,255),thickness=2)
		
	cv2.rectangle(frame,(500,0),(540,40),(b,g,r),-1)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		print(allpts)
		break
	elif key == ord("r"):
		allpts=[]
	elif key == ord("h"):
		if helpme ==True:
			helpme = False
		else:
			helpme = True
	elif key == ord("["):
		if thickness>1:
			thickness-=1
	elif key == ord("]"):
		thickness+=1
	elif key == ord("u"):
		allpts = allpts[:-1]
	elif key == ord("s"):
		cv2.imwrite(r"Drawings\drawing%i.jpg"%savecount,frame)
		savecount+=1
	elif key == ord("p"):
		cv2.imwrite(r"Drawings\drawing%i.jpg"%savecount,paint)
		savecount+=1
	#print(allpts)
	#print(pts)
	
# if we are not using a video file, stop the camera video stream
vs.release()
# close all windows
cv2.destroyAllWindows()