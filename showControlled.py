#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:07:04 2018

@author: kiyoshitaro
"""

import cv2
import numpy as np
import glob
import time

def load(path):
    """takes as input the path to a .pts and returns a list of 
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return points

path = './data/use/*.pts'
imgs = []
paths = glob.glob(path)
for path in paths:
    imgs.append(load(path))
    
imgs = np.floor(np.array(imgs))
imgs = imgs.astype(int)



zeros = np.zeros((500,500,3),np.uint8)
for img in imgs:
    for (x,y) in img:
        zeros = cv2.circle(zeros,(x,y),1,(0,0,255),0)
        
    cv2.imwrite('./result/images/' + str(time.time()) + '.jpg' ,zeros)
    zeros = np.zeros((500,500,3),np.uint8)