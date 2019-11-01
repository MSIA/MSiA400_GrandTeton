#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:46:54 2019

@author: wangjue
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def count_contours(img):
    image = cv2.imread(img)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray,30,200)
    _, contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count_contours = len(contours)
    return count_contours

def count_squares(img):
    image = cv2.imread(img)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh=cv2.threshold(gray,127,255,1)
    _,contours,hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    count_squares = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            x,y,w,h=cv2.boundingRect(cnt)
            M=cv2.moments(cnt)
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
                #cv2.boundingRect return the left width and height in pixels, starting from the top
            #left corner, for square it would be roughly same     
            if abs(w-h) <= 3:
                count_squares += 1
                shape_name="square"
                #find contour center to place text at center
                cv2.drawContours(image,[cnt],0,(0,125,255),-1)
                cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)         
            else:
                shape_name="Reactangle"
                #find contour center to place text at center
                cv2.drawContours(image,[cnt],0,(0,0,255),-1)
                M=cv2.moments(cnt)
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    return count_squares