#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath,do_preprocess = True):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    if do_preprocess:
        prob = np.random.uniform(0.0,1.0)
        if prob >=0.6:  # 40% probability of data augmentation- We are not changing the number of samples 
            I1 = AugmentImages(I1)

        # Standardization
        I1 = cv2.normalize(I1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred,ModelName):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    print('Loading: ' , ModelName)
    if ModelName == 'BasicModel':
        do_preprocess  = False

        _, prSoftMaxS = BasicModel(ImgPH)   #---------------------------------------------------------Change Model
        
    if ModelName == 'BasicModel2':
        do_preprocess  = True
        _, prSoftMaxS = BasicModel2(ImgPH)   #---------------------------------------------------------Change Model

    if ModelName == 'ResNet':
        do_preprocess  = True
        _, prSoftMaxS = ResNet(ImgPH)
    if ModelName == 'ResNet2':
        do_preprocess  = True
        _, prSoftMaxS = ResNet2(ImgPH)
    if ModelName == 'DenseNet':
        do_preprocess  = True
        _, prSoftMaxS = DenseNet(ImgPH)
    if ModelName == 'ResNext':
        do_preprocess  = True
        _, prSoftMaxS = ResNext(ImgPH)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        OutSaveT = open(LabelsPathPred, 'w')

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow, do_preprocess)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

            OutSaveT.write(str(PredT)+'\n')
            
        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = list(map(float, LabelTest.split()))

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = list(map(float, LabelPred.split()))
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))
    
    accuracy = Accuracy(LabelsPred, LabelsTrue)
    print('Accuracy: '+ str(accuracy), '%')
    
    OutFile = './Results.txt'
    f = open(OutFile, "a")
    f.write(str(cm) + '\n' + 'Accuracy: '+ str(accuracy) + '% \n')
    f.close()


        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelName', default='BasicModel', help='Call the model name, Default= ResNext')#--------------------------------Change LogsPath
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/BasicModel/14model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelName = Args.ModelName
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    
    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred,ModelName)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
