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
# termcolor, do (pip install termcolor)

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

if type(tf.contrib) != type(tf): tf.contrib._warning = None


# Don't generate pyc codes
sys.dont_write_bytecode = True
    
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,do_preprocess = True):
    """
    Inputs: 
    BasePath - Path to CIFAR10 folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.png'   
        ImageNum += 1
    	
        I1 = cv2.imread(RandImageName)
        Label = convertToOneHot(TrainLabels[RandIdx], 10)
        
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################
        if do_preprocess:
            prob = np.random.uniform(0.0,1.0)
            if prob >=0.6:  # 40% probability of data augmentation- We are not changing the number of samples 
                I1 = AugmentImages(I1)

            # Standardization
            I1 = cv2.normalize(I1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(Label)
        
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelName):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to CIFAR10 folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    
    # Predict output with forward pass
    print('Loading: ' , ModelName)
    if ModelName == 'BasicModel':
        do_preprocess  = False

        prLogits, prSoftMax = BasicModel(ImgPH)   #---------------------------------------------------------Change Model
        
    if ModelName == 'BasicModel2':
        do_preprocess  = True
        prLogits, prSoftMax = BasicModel2(ImgPH)   #---------------------------------------------------------Change Model

    if ModelName == 'ResNet':
        do_preprocess  = True
        prLogits, prSoftMax = ResNet(ImgPH)
    if ModelName == 'ResNet2':
        do_preprocess  = True
        prLogits, prSoftMax = ResNet2(ImgPH)
    if ModelName == 'DenseNet':
        do_preprocess  = True
        prLogits, prSoftMax = DenseNet(ImgPH)
    if ModelName == 'ResNext':
        do_preprocess  = True
        prLogits, prSoftMax = ResNext(ImgPH)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = LabelPH,logits = prLogits))

    with tf.name_scope('Accuracy'):
        prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
        LabelDecoded = tf.argmax(LabelPH, axis=1)
        Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
        
    with tf.name_scope('Adam'):
    	###############################################
    	# Fill your optimizer of choice here!
    	###############################################
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.scalar('Accuracy', Acc)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    
    ## Load Test Set
    DirNamesTest,TestLabels,NumTestSamples = SetupAll(BasePath, CheckPointPath,False)

    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
        saveLoss = []
        saveTrainAcc = []
        saveTestAcc = []
        
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                
                # generate Train Data Batch
                TrainI1Batch, TrainLabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,do_preprocess)
                FeedDict = {ImgPH: TrainI1Batch, LabelPH: TrainLabelBatch}
                
                # generate Test Data Batch
                TestI1Batch, TestLabelBatch = GenerateBatch(BasePath, DirNamesTest, TestLabels, ImageSize, MiniBatchSize,do_preprocess)
                Test_FeedDict = {ImgPH: TestI1Batch, LabelPH: TestLabelBatch}
                
                # Train
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                
                # compute Train and Test Accuracy
                TrainAccuracy = sess.run(Acc,feed_dict=FeedDict)
                TestAccuracy = sess.run(Acc,feed_dict=Test_FeedDict)
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')
                    print('\n Loss : ', LossThisBatch,'\t TrainAcc : ', TrainAccuracy,'\t TestAcc : ', TestAccuracy)
                    
                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
            
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            
            saveLoss.append(LossThisBatch) # every epoch
            saveTrainAcc.append(TrainAccuracy) # every epoch
            saveTestAcc.append(TestAccuracy) # every epoch

        np.save(LogsPath+'Loss'+ModelName, np.array(saveLoss))
        np.save(LogsPath+'TrainAcc'+ModelName, np.array(saveTrainAcc))
        np.save(LogsPath+'TestAcc'+ModelName, np.array(saveTestAcc))

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../CIFAR10', help='Base path of images, Default:../CIFAR10')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/ResNext/', help='Path to save Checkpoints, Default: ../Checkpoints/') #--- Change CheckPointsPath
    Parser.add_argument('--NumEpochs', type=int, default=25, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/ResNext/', help='Path to save Logs for Tensorboard, Default=Logs/')#--------------------------------Change LogsPath
    Parser.add_argument('--ModelName', default='ResNext', help='Call the model name, Default= ResNext')#--------------------------------Change Model
    Args = Parser.parse_args()
    NumEpochs = int(Args.NumEpochs)
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = int(Args.MiniBatchSize)
    LoadCheckPoint = int(Args.LoadCheckPoint)
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelName = Args.ModelName
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath,True)
    
    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
    
    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelName)
    
if __name__ == '__main__':
    main()
 
