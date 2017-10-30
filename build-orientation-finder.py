# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
from colour import Color

# solar range in degrees
solar_range = 30

#minimum house area
min_area =20
 
# house color in google maps
# rgb(238,238,238)
low = 235
high = 242

lower_house = np.array([low,low,low])
upper_house = np.array([high,high,high])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
 
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread("./data/"+args["image"])
house = cv2.inRange(image,lower_house,upper_house)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find contours in the thresholded image
cnts = cv2.findContours(house.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

house = cv2.merge([house, house, house])
contours = cv2.merge([gray, gray, gray])
boxes = contours.copy()
arrows = contours.copy()
shaded = contours.copy()

(h, w) = image.shape[:2]
output = np.zeros((h * 6, w,3), dtype="uint8")
output[0:h,     0:w] = image
output[h:2*h,   0:w] = house
output[2*h:3*h, 0:w] = contours
output[3*h:4*h, 0:w] = boxes
output[4*h:5*h, 0:w] = arrows
output[5*h:6*h, 0:w] = shaded

# loop over the contours
for c in cnts:
	area = cv2.contourArea(c)

	# if the contour is too small, ignore it
	if area < min_area:
		continue

	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the contour and center of the shape on the image
	cv2.drawContours(contours, [c], -1, (255, 0, 150), 2)
	#cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
	#cv2.putText(image, "center", (cX - 20, cY - 20),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	center=rect[0]

	angle_deg = rect[2]
	dim = rect[1]
	rect_w = dim[0]
	rect_h = dim[1]

	if (rect_w < rect_h):
  		angle_deg = 90 + angle_deg

  	angle_deg = 90 + angle_deg

	angle = np.radians(angle_deg)
	cv2.drawContours(boxes,[box],0,(255,0,0),2)

	length = 20
	end_x = int(center[0]+(length*np.cos(angle)))
	end_y = int(center[1]+(length*np.sin(angle)))
	endpoint=(end_x,end_y)
	center=(int(center[0]),int(center[1]))

	red=Color("red")
	yellow=Color("yellow")
	colors = list(red.range_to(yellow,91))

	#get distance from south
	distance = min(abs(90-angle_deg),90)

	arrow_color = colors[int(distance)]
	R = int(255*arrow_color.rgb[2])
	G = int(255*arrow_color.rgb[1])
	B = int(255*arrow_color.rgb[0])

	cv2.arrowedLine(arrows,center,endpoint,(R,G,B),2,tipLength=0.3)

	#if(int(angle_deg) >(90-solar_range) and int(angle_deg)<(90+solar_range)):
	#	cv2.arrowedLine(arrows,center,endpoint,(0,255,0),2,tipLength=0.3)
	#else:
	#	cv2.arrowedLine(arrows,center,endpoint,(255,0,0),2,tipLength=0.3)

	if(int(angle_deg) >(90-solar_range) and int(angle_deg)<(90+solar_range)):
		cv2.drawContours(shaded, [c], 0, (100, 220, 100), -1)

	# show the image
	output[0:h,     0:w] = image
	output[h:2*h,   0:w] = house
	output[2*h:3*h, 0:w] = contours
	output[3*h:4*h, 0:w] = boxes
	output[4*h:5*h, 0:w] = arrows
	output[5*h:6*h, 0:w] = shaded
	#cv2.imshow("Test", output)
	cv2.imwrite("./output/SOLAR-PLOT-ALL"+args["image"],output)
	cv2.imwrite("./output/SOLAR-PLOT-1"+args["image"],house)
	cv2.imwrite("./output/SOLAR-PLOT-2"+args["image"],contours)
	cv2.imwrite("./output/SOLAR-PLOT-3"+args["image"],boxes)
	cv2.imwrite("./output/SOLAR-PLOT-4"+args["image"],arrows)
	cv2.imwrite("./output/SOLAR-PLOT-5"+args["image"],shaded)

