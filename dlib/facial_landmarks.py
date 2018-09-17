# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import time



#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#image = cv2.imread(args["image"])

#link = ./images/sample_data_2/*.jpg
link = '../../../../Pictures/Data/*.png'
listurls = glob.glob(link)
for url in listurls :
    image = cv2.imread(url)
    image = imutils.resize(image, width=500)
#    cv2.imwrite('after imutils.resize.jpg',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    cv2.imwrite('after to gray.jpg', gray)
    
    rects = detector(gray, 1)
    
    for (i, rect) in enumerate(rects):
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
    
    	(x, y, w, h) = face_utils.rect_to_bb(rect)
    	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    	for (x, y) in shape:
    		cv2.circle(image, (x, y), 1, (0, 0, 255), 0)
    
    cv2.imwrite("../../../../Pictures/result/" + str(time.time()) +".jpg",image)
#cv2.waitKey(0)
