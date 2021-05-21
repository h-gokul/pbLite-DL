#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
from helper_functions import *
import math
import matplotlib.pyplot as plt
import imutils
from sklearn.cluster import KMeans
import scipy.stats as st

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImageName', default='7.jpg', help='Image name, Default:7.jpg')


    Args = Parser.parse_args()
    imName = Args.ImageName
    im = cv2.imread('../BSDS500/Images/' + str(imName))
    imGray = im[:,:,0]


    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    size  = 15
    scales = [1,2]
    n_orientations = 8

    OrientedDoGFilterBank = OrientedDoG(size,scales,n_orientations)
    n_filters = OrientedDoGFilterBank.shape[2]
    
    Fig1 = plt.figure()
    for i in range(n_filters):
        ax = Fig1.add_subplot(2, 10, i+1)			
        plt.imshow(OrientedDoGFilterBank[:,:,i],'gray')
        ax.set_xticks([])
        ax.set_yticks([])
    

    Fig1.suptitle("DoG Filter Bank", fontsize=20, fontweight = 'bold')
    plt.savefig('../Outputs/DoG.png')	

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    size = 29
    LMFilterBank = LM_FilterBank(size)
    n_filters = LMFilterBank.shape[2]

    Fig1 = plt.figure()
    for i in range(n_filters):
        ax = Fig1.add_subplot(3, 16, i+1)
        plt.imshow(LMFilterBank[:,:,i],'gray')
        ax.set_xticks([])
        ax.set_yticks([])

    Fig1.suptitle("LM Filter Bank", fontsize=20, fontweight = 'bold')
    plt.savefig('../Outputs/LM.png')	



    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    size = 29
    params = [[4,1.2*np.pi,1],[5,1.5*np.pi,1],[6,1.9*np.pi,1],[7,3*np.pi,0.7],[11,5*np.pi,0.7]]

    gaborFilterBank = GaborFilterBank(size,params,n_orientations = 8)
    
    n_filters = gaborFilterBank.shape[2]

    Fig1 = plt.figure()
    for i in range(n_filters):
        ax = Fig1.add_subplot(6, 16, i+1)
        plt.imshow(gaborFilterBank[:,:,i],'gray')
        ax.set_xticks([])
        ax.set_yticks([])

    Fig1.suptitle("gabor Filter Bank", fontsize=20, fontweight = 'bold')
    plt.savefig('../Outputs/Gabor.png')	
    print('Done FilterBanks')

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    size1,size2,size3 = 11,21,31
    n_Diskorientations = 16
    HalfDisk1,HalfDisk2,HalfDisk3 = getHalfDisks(size1,size2,size3,n_Diskorientations = n_Diskorientations)

    HD = [HalfDisk1[:,:,i] for i in  range(n_Diskorientations)]
    for i in  range(n_Diskorientations):
        HD.append(HalfDisk2[:,:,i])

    for i in  range(n_Diskorientations):
        HD.append(HalfDisk3[:,:,i])

    Fig1 = plt.figure()
    for i in range(len(HD)):
        ax = Fig1.add_subplot(6, 8, i+1)
        plt.imshow(HD[i],'gray')
        ax.set_xticks([])
        ax.set_yticks([])

    Fig1.suptitle("Half Disk Masks", fontsize=20, fontweight = 'bold')
    plt.savefig('../Outputs/HDMasks.png')
    print('Done HalfDisks')
    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """
    O_response = getFilterResponse(imGray,OrientedDoGFilterBank)
    L_response = getFilterResponse(imGray,LMFilterBank)
    G_response = getFilterResponse(imGray,gaborFilterBank)
    FilterResponse = np.dstack((O_response,G_response,L_response))


    """
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """
    Tmap = do_Kmeans(O_response,K= 64 ) # takes a lot of time for generating resposne with all 102 filters in 3 filter banks 

    fig = plt.figure()
    plt.imshow(Tmap)
    imNameSplit = imName.split('.')
    fileName = "../Outputs/TextonMap_" + imNameSplit[0] + ".png"
    fig.suptitle("Texton Map", fontsize=20)
    plt.savefig(fileName)

    """
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    num_bins = 64 
    DiskBanks = [HalfDisk1,HalfDisk2,HalfDisk3]
    n_Diskpairs = 8

    T_g = getGradient(Tmap,num_bins,DiskBanks,n_Diskpairs)

    fig = plt.figure()
    plt.imshow(T_g)
    imNameSplit = imName.split('.')
    fileName = "../Outputs/Tg_" + imNameSplit[0] + ".png"
    fig.suptitle("Texton Gradient", fontsize=20)
    plt.savefig(fileName)
    print('Done T_g')
    """
    Generate Brightness Map
    Perform brightness binning 
    """
    Bmap = do_Kmeans(imGray,K= 16)

    """
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
    """    
    B_g = getGradient(Bmap,num_bins,DiskBanks,n_Diskpairs)

    fig = plt.figure()
    plt.imshow(B_g)
    imNameSplit = imName.split('.')
    fileName = "../Outputs/Bg_" + imNameSplit[0] + ".png"
    fig.suptitle("Brightness Gradient", fontsize=20)
    plt.savefig(fileName)
    print('Done B_g')


    """
    Generate Color Map
    Perform color binning or clustering
    """
    Cmap = do_Kmeans(im,K=16) 
    
    """
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    C_g = getGradient(Cmap,num_bins,DiskBanks,n_Diskpairs)

    fig = plt.figure()
    plt.imshow(C_g)
    imNameSplit = imName.split('.')
    fileName = "../Outputs/Cg_" + imNameSplit[0] + ".png"
    fig.suptitle("Color Gradient", fontsize=20)
    plt.savefig(fileName)
    print('Done C_g')
    """
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """
    path = "../BSDS500/SobelBaseline/" + imNameSplit[0] + ".png"
    Sobel = cv2.imread(path, 0)

    """
    Read Canny Baseline
    use command "cv2.imread(...)"
    """
    path = "../BSDS500/SobelBaseline/" + imNameSplit[0] + ".png"
    Canny = cv2.imread(path, 0)


    """
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """

    A = (T_g+B_g+C_g)/3
    
    w1,w2 = 0.5,0.5
    B = w1*Canny+w2*Sobel
    pb = np.multiply(A, B)
    cv2.normalize(pb, pb, 0, 255, cv2.NORM_MINMAX)
    print('Done pbLite')
    fig = plt.figure()
    plt.imshow(C_g)
    imNameSplit = imName.split('.')
    fileName = "../Outputs/PbLite" + imNameSplit[0] + ".png"
    fig.suptitle("PbLite Result", fontsize=20)
    plt.savefig(fileName)
    
if __name__ == '__main__':
    main()
 