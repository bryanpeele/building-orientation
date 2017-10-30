# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
from colour import Color

# "optimal" solar range in degrees from due south
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
 
# load the image, threshold by colot, save a grayscale copy of original
image = cv2.imread("./data/"+args["image"])
house = cv2.inRange(image,lower_house,upper_house)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find contours in the thresholded image
cnts = cv2.findContours(house.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# setup each scene for plotting on top of later
house = cv2.merge([house, house, house])
contours = cv2.merge([gray, gray, gray])
boxes = contours.copy()
arrows = contours.copy()
shaded = contours.copy()

# define array to hold cummulative image with all six steps
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
	# find area of selected contour
	area = cv2.contourArea(c)

	# if the contour is too small, ignore it
	if area < min_area:
		continue

	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the contour and center of the shape on the image (purple)
	cv2.drawContours(contours, [c], -1, (255, 0, 150), 2)

	# for each contour, fit a rectangle
	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	center=rect[0]

	#Store center, angle, width and height for fit rectangle
	angle_deg = rect[2]
	dim = rect[1]
	rect_w = dim[0]
	rect_h = dim[1]

	# correct angle to make sure follows major axis
	if (rect_w < rect_h):
  		angle_deg = 90 + angle_deg

  	# convert to minor axis
  	angle_deg = 90 + angle_deg

  	#convert to radians
	angle = np.radians(angle_deg)

	#plot fit rectangles (in blue)
	cv2.drawContours(boxes,[box],0,(255,0,0),2)

	# provide length and calculate endpoints for arrow pointing
	# in optimal solar direction for each building
	length = 20
	end_x = int(center[0]+(length*np.cos(angle)))
	end_y = int(center[1]+(length*np.sin(angle)))
	endpoint=(end_x,end_y)
	center=(int(center[0]),int(center[1]))

	# set up heatmap from red (best) to yellow (worst)
	# colors[0] == red, colors[90] == yellow 
	red=Color("red")
	yellow=Color("yellow")
	colors = list(red.range_to(yellow,91))

	#get angular distance from south
	distance = min(abs(90-angle_deg),90)

	#get arrow color and output as integer RGB values
	arrow_color = colors[int(distance)]
	R = int(255*arrow_color.rgb[2])
	G = int(255*arrow_color.rgb[1])
	B = int(255*arrow_color.rgb[0])

	# draw arrows
	cv2.arrowedLine(arrows,center,endpoint,(R,G,B),2,tipLength=0.3)

	# select best candidate buildings and highlight in green
	if(int(angle_deg) >(90-solar_range) and int(angle_deg)<(90+solar_range)):
		cv2.drawContours(shaded, [c], 0, (100, 220, 100), -1)

	# store output with all images/plots
	output[0:h,     0:w] = image
	output[h:2*h,   0:w] = house
	output[2*h:3*h, 0:w] = contours
	output[3*h:4*h, 0:w] = boxes
	output[4*h:5*h, 0:w] = arrows
	output[5*h:6*h, 0:w] = shaded
	
	# Output PNG for each image/plot individually as well as
	# combined image
	cv2.imwrite("./output/SOLAR-PLOT-ALL"+args["image"],output)
	cv2.imwrite("./output/SOLAR-PLOT-1"+args["image"],house)
	cv2.imwrite("./output/SOLAR-PLOT-2"+args["image"],contours)
	cv2.imwrite("./output/SOLAR-PLOT-3"+args["image"],boxes)
	cv2.imwrite("./output/SOLAR-PLOT-4"+args["image"],arrows)
	cv2.imwrite("./output/SOLAR-PLOT-5"+args["image"],shaded)

