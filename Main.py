#!/usr/bin/env python

import cv2
import numpy as np
import os
import time 
import DetectChars
import DetectPlates
import PossiblePlate

cap = cv2.VideoCapture(0)


index_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
filter_list = ["No change","N","N","N","N","N","N","N","N","N","No change","N","N","N","N","N","N","N","N","N"] 
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)




def main_detect (imgOriginalScene, k):
	
	listOfPossiblePlates=DetectPlates.detectPlatesInScene(imgOriginalScene)
	listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)          
		
	if len(listOfPossiblePlates) == 0: 
		return 0
	else:                                                  
		listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
		licPlate = listOfPossiblePlates[0]
	
		len_plate = len (licPlate.strChars)
		
		if len_plate < 9 and len_plate>0:  
			if filter_list.count(licPlate)>0 :
				index_list[k] +=1 
			else :
				filter_list[k] = licPlate
				index_list[k] = k
				k+=1



def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function


if __name__ == "__main__":
	blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training
	if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
		print ("\nerror: KNN traning was not successful\n" ) 
	while(True):
		ret, frame = cap.read()
		imgOriginalScene = frame[0:240, 0:360]		
		k = 0
		cv2.imshow("ImgOriginalScene", imgOriginalScene)
		for i in range(0,20):
			main_detect(imgOriginalScene,k)

		if (filter_list[0] != "No change") :	
			licPlate = filter_list[index_list.index(max(index_list))] 	  
			drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate
			print ("\nlicense plate read from image = " + licPlate.strChars + "\n" )      # write license plate text to std out
			writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image
			cv2.imshow("ImgOriginalScene", imgOriginalScene)
			filter_list[0] = "No change"

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	

















