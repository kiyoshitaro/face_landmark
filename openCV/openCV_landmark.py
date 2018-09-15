#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:43:20 2018

@author: kiyoshitaro
"""
from imutils import face_utils
import cv2
import imutils
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

url = './images/sample_data_2/'
img = cv2.imread('2018-09-14-104330.jpg') 

#img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.2,5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    for(ex,ey, ew, eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex + ew, ey + eh),(0,255,0),2)
        
cv2.imwrite('test.png', img)
        