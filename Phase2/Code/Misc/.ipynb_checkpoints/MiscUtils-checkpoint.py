"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Don't generate pyc codes
sys.dont_write_bytecode = True

def StandardizeInputs(img):
	img /= 255
	img -= 0.5
	img *= 2

def tic():
    """
    Function to start timer
    Tries to mimic tic() toc() in MATLAB
    """
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    """
    Function to stop timer
    Tries to mimic tic() toc() in MATLAB
    """
    return time.time() - StartTime

def FindLatestModel(CheckPointPath):
    """
    Finds Latest Model in CheckPointPath
    Inputs:
    CheckPointPath - Path where you have stored checkpoints
    Outputs:
    LatestFile - File Name of the latest checkpoint
    """
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile


def convertToOneHot(vector, NumClasses):
    """
    vector - vector of argmax indexes
    NumClasses - Number of classes
    """
    return np.equal.outer(vector, np.arange(NumClasses)).astype(np.float)


def AugmentImages(img):  
    # change color format
    # code = np.random.randint(0, 5) # more probability for image to be sent as it is
    # if code == 0:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # if code == 1:
    #     img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # if code == 2:
    #     img = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    # if code == 3:
    #     img = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    # if code == 4:
    #     img = cv2.cvtColor(im, cv2.COLOR_BGR2LUV)

    # flip
    code = np.random.randint(-2, 2) # more probability for image to be sent as it is
    if code > -2:
        img = cv2.flip(img, code) 

    # change brightness, contrast
    """
    alpha 1  beta 0      --> no change  
    0 < alpha < 1        --> lower contrast  
    alpha > 1            --> higher contrast  
    -127 < beta < +127   --> good range for brightness values
    """
    beta = int(np.random.uniform(-50, 50))
    alpha = 1 + np.random.uniform(-0.2, 0.2)
    img = alpha*img + beta

    # change noise
    noise = np.zeros(img.shape, np.uint16)
    noise = cv2.randn(noise, 0, 15)
    img = cv2.add(img, noise, dtype=cv2.CV_8UC3)
    return img

