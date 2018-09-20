#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:11:56 2018

@author: kiyoshitaro
"""
#
#import numpy as np
#import cv2
#
#
#
#img = cv2.imread('./data/image1.jpg')
#
#cv2.imshow('image', img)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
import dlib
import sys
from skimage import io
import glob
import time

link = './sample_data_2/*2018-09-20-222725.jpg'
listurls = glob.glob(link)
detector = dlib.get_frontal_face_detector()

for image_path in listurls:
#    image_path = '../../../Pictures/Data/image_0003.png'
    img = io.imread(image_path)
    win = dlib.image_window()
    
    
    dets = detector(img, 1) 
    win.clear_overlay()
    
    win.add_overlay(dets) 
    win.set_image(img)
    
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    for k, d in enumerate(dets):
        # Xác định facial landmark trên khuôn mặt thánh nữ
        shape = predictor(img, d)
        # Vẽ facial landmark lên bức ảnh
        win.add_overlay(shape)
    
    
#    io.imsave("./result/" + str(time.time()) +".jpg",img)


