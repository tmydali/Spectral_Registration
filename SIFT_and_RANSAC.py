#!/usr/bin/env python
# coding: utf-8

# -------------------- #
# This is a multiprocessing version which can't run on notebook
# -------------------- #

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, isdir, join
from multiprocessing import Pool

def readImg(src):
    head = cv2.imread(src[0], cv2.IMREAD_GRAYSCALE)
    img = np.zeros((head.shape[0], head.shape[1], 4), dtype='uint8')

    img[:,:,0] = head
    img[:,:,1] = cv2.imread(src[1], cv2.IMREAD_GRAYSCALE)
    img[:,:,2] = cv2.imread(src[2], cv2.IMREAD_GRAYSCALE)
    img[:,:,3] = cv2.imread(src[3], cv2.IMREAD_GRAYSCALE)

    return img

# Brute-Force Matching with SIFT Descriptors

def SIFT(img):
    siftDetector= cv2.xfeatures2d.SIFT_create()
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def matcher(des1, des2): 
    # create Matcher object
    bf_matcher = cv2.BFMatcher()

    # Match descriptors.
    matches = bf_matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x: x.distance)
    return matches

def getHomography(matches, kp1, kp2):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors

    #h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    h, mask = cv2.estimateAffinePartial2D(points2, points1)
    return h

def register(img):
    for i in [1,2,3]:
        # Find matching points
        kp1, des1 = SIFT(img[:,:,0])
        kp2, des2 = SIFT(img[:,:,i])
        matches = matcher(des1, des2)

        # Use homography
        h = getHomography(matches, kp1, kp2)
        height, width, channels = img.shape
        # img[:,:,i] = cv2.warpPerspective(img[:,:,i], h, (width, height))  #Applies a perspective transformation to an image.
        img[:,:,i] = cv2.warpAffine(img[:,:,i], h, (width, height))

        # print("Estimated homography : \n",  h)
    return img

def saveImg(img, f, tPath):
    fullpath = join(tPath, f)
    os.mkdir(fullpath)
    RGBpath = join(fullpath, 'RGB.tif')
    FalseColorpath = join(fullpath, 'FalseColor.tif')
    cv2.imwrite(RGBpath, img[:, :, :3])
    cv2.imwrite(FalseColorpath, img[:, :, 1:4])
    print('Image "{}" saved'.format(f))
    
def process(f): 
    dataPath = 'C:/Users/tmyda/Documents/UAV_color'
    targetPath = 'C:/Users/tmyda/Documents/SIFT'
    
    fullpath = join(dataPath, f)
    if isfile(fullpath): pass
    elif isdir(fullpath):
        srcImg = listdir(fullpath)
        for index, item in enumerate(srcImg):
            srcImg[index] = join(fullpath, item)
        img = readImg(srcImg)
        register(img)
        saveImg(img, f, targetPath)
    
    return

if __name__ == '__main__':
    dataPath = 'C:/Users/tmyda/Documents/UAV_color'
    targetPath = 'C:/Users/tmyda/Documents/SIFT'
    os.mkdir(targetPath)
    files = listdir(dataPath)
    
    with Pool(10) as pool:
    	pool.map(process, files)
    