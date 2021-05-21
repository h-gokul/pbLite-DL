"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

import warnings
warnings.filterwarnings('ignore')

if type(tf.contrib) != type(tf): tf.contrib._warning = None

def BasicModel(Img,num_classes = 10):
    
    """ Only 2 Conv Layers and 2 Dense """
   
    model = tf.layers.conv2d(inputs = Img, filters=32,kernel_size=3, padding='same',activation = tf.nn.relu)
    
    model = tf.layers.conv2d(model, filters=64,kernel_size=3, padding='same',activation = tf.nn.relu)
    
    model = tf.layers.flatten(model)
    
    model = tf.layers.dense(inputs = model, units = 128, activation = tf.nn.relu)

    prLogits = tf.layers.dense(inputs = model,units=num_classes, activation=None)
    
    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)
    
    return prLogits, prSoftMax

def BasicModel2(Img,num_classes = 10):
    
    """  5 x Conv + MaxPool+ BatchNorm 
    and 3 x Dense """
        
    model = tf.layers.conv2d(inputs = Img, padding='SAME',filters = 32, kernel_size = 3, activation = tf.nn.relu)
    model = tf.nn.max_pool(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.conv2d(inputs = model, padding='SAME',filters = 64, kernel_size = 3, activation = tf.nn.relu)
    model = tf.nn.max_pool(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.conv2d(inputs = model, padding='SAME',filters = 128, kernel_size = 3, activation = tf.nn.relu)
    model = tf.nn.max_pool(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    model = tf.layers.batch_normalization(model)

    model = tf.layers.conv2d(inputs = model, padding='SAME',filters = 256, kernel_size = 3, activation = tf.nn.relu)
    model = tf.nn.max_pool(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.conv2d(inputs = model, padding='SAME',filters = 512, kernel_size = 3, activation = tf.nn.relu)
    model = tf.nn.max_pool(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.flatten(model)
    
    model = tf.layers.dense(inputs = model, units = 128, activation = tf.nn.relu)
    model = tf.nn.dropout(model, 0.7)
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.dense(inputs = model, units = 256, activation = tf.nn.relu)
    model = tf.nn.dropout(model, 0.7)
    model = tf.layers.batch_normalization(model)

    model = tf.layers.dense(inputs = model, units = 512, activation = tf.nn.relu)
    model = tf.nn.dropout(model, 0.7)
    model = tf.layers.batch_normalization(model)

    prLogits = tf.layers.dense(inputs = model,units=num_classes, activation=None)    
    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)

    return prLogits, prSoftMax

######################################################### RESNET #########################################################
"""
ResNet20 V2 Implementation: 
    from Identity Mappings in Deep Residual Networks (https://arxiv.org/pdf/1603.05027.pdf)

Model Block Diagram Referred from:
    https://github.com/chao-ji/tf-resnet-cifar10/blob/master/files/ResNet20_CIFAR10.pdf
"""

def identityUnit(model,n_units,n_blocks):
    
    if ((n_blocks ==2)&(n_units ==1)):
        model = tf.nn.avg_pool2d(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        return tf.pad(model, [[0, 0], [0, 0], [0, 0], [8,8]])

    elif ((n_blocks ==3)&(n_units ==1)):
        model = tf.nn.avg_pool2d(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        return tf.pad(model, [[0, 0], [0, 0], [0, 0], [16,16]])
    
    else:
        return model
    
def basicUnit(model,filters, n_units, n_blocks):
    print("ENTERED UNIT: ", n_units, "  ", n_blocks)

    if ((n_units == 1)&(n_blocks > 1)):   # to downsample on first layer of every block
        print('STRIDE CHANGED')
        strides = (2, 2)
    else:
        strides = (1, 1)
        
    I = identityUnit( model, n_units, n_blocks) # skip connection is taken in the beginning, by default

    ##### Layer 1 ###########################################################################
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    
    if ((n_units == 1)&(n_blocks == 1)) :  # get skip connection here only if it is the first unit of first block
        strides = (1,1)  # to avoid downsampling first unit of first block
        print('pre-activation skip connection accessed for 1st unit 1st block')
        I = identityUnit(model=model, n_units=1, n_blocks=1)
        
    model = tf.layers.conv2d(inputs = model, filters=filters,kernel_size=3,strides=strides, padding='SAME', activation = None)
    
    ##### Layer 2 ###########################################################################
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    model = tf.layers.conv2d(inputs = model, filters=filters,kernel_size=3, strides = (1,1), padding='SAME', activation = None)
    
    print(" ADDING MODEL and I")
    model = tf.math.add(I,model)
    
    return model

def resBlock(model,filters,n_blocks):
    print( '\n ResNet Block: '+ str(n_blocks),' \t No. of filters: '+ str(filters))
    print( '\n \t ResNet Unit: '+ str(1))
    model = basicUnit(model=model, filters=filters, n_units = 1, n_blocks = n_blocks)
    print( '\n \t ResNet Unit: '+ str(2))
    model = basicUnit(model=model, filters=filters, n_units = 2, n_blocks = n_blocks)
    print( '\n \t ResNet Unit: '+ str(3))
    model = basicUnit(model=model, filters=filters, n_units = 3, n_blocks = n_blocks)
    
    return model
    
def ResNet(Img,num_classes=10):
    
    model = tf.layers.conv2d(inputs = Img, filters=16,kernel_size=3, padding='SAME', activation = None)
    
    model = resBlock(model,filters=16,n_blocks=1)
    model = resBlock(model,filters=32,n_blocks=2)
    model = resBlock(model,filters=64,n_blocks=3)
        
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    
    model = tf.layers.flatten(model)
    
    model = tf.layers.dense(inputs = model, units = 128, activation = tf.nn.relu)

    prLogits = tf.layers.dense(inputs = model,units=num_classes, activation=None)
    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)
    
    return prLogits, prSoftMax


######################################################### RESNET V1 #########################################################
"""
Simple ResNet20 Implementation:  (Removed Downsampling (AvgPooling & Strides) and  pre-Activation)
     Deep Residual Learning for Image Recognition: (https://arxiv.org/abs/1512.03385)
     Identity Mappings in Deep Residual Networks: (https://arxiv.org/pdf/1603.05027.pdf)
     

Model Block Diagram Referred from:
    https://github.com/chao-ji/tf-resnet-cifar10/blob/master/files/ResNet20_CIFAR10.pdf

"""
    
def basicUnit2(model,filters, n_units, n_blocks):
        
    I = model # skip connection is taken in the beginning, by default

    ##### Layer 1 ###########################################################################
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    model = tf.layers.conv2d(inputs = model, filters=filters,kernel_size=3, padding='SAME', activation = None)
    
    ##### Layer 2 ###########################################################################
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    model = tf.layers.conv2d(inputs = model, filters=filters,kernel_size=3, padding='SAME', activation = None)
    
    print(" ADDING MODEL and I")
    model = tf.math.add(I,model)
    
    return model

def resBlock2(model,filters,n_blocks):
    print( '\n ResNet Block: '+ str(n_blocks),' \t No. of filters: '+ str(filters))
    print( '\n \t ResNet Unit: '+ str(1))
    model = basicUnit(model=model, filters=filters, n_units = 1, n_blocks = n_blocks)
    print( '\n \t ResNet Unit: '+ str(2))
    model = basicUnit(model=model, filters=filters, n_units = 2, n_blocks = n_blocks)
    print( '\n \t ResNet Unit: '+ str(3))
    model = basicUnit(model=model, filters=filters, n_units = 3, n_blocks = n_blocks)
    
    return model
    
def ResNet2(Img,num_classes=10):
    
    model = tf.layers.conv2d(inputs = Img, filters=16,kernel_size=3, padding='SAME', activation = None)
    
    model = resBlock(model,filters=16,n_blocks=1)
    model = resBlock(model,filters=32,n_blocks=2)
    model = resBlock(model,filters=64,n_blocks=3)
        
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    
    model = tf.layers.flatten(model)
    
    model = tf.layers.dense(inputs = model, units = 128, activation = tf.nn.relu)

    prLogits = tf.layers.dense(inputs = model,units=num_classes, activation=None)
    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)
    
    return prLogits, prSoftMax

##############################################################################################
"""
Simple DesNet Implementation:  compressed in both transition and dense blocks
     Deep Residual Learning for Image Recognition: (https://arxiv.org/pdf/1608.06993.pdf)   

Input --> Conv2D + DenseBlock_3 + TransistionBlock + DenseBlock_3 + (Bn+Relu+AvgPooling) + Dense --> Logits

"""
def DenseBlock(model, n_filters = 36, dropout = 0.2, l=6, compression = 0.5):
    tmp = model
    
    for _ in range(l):
        model = tf.layers.batch_normalization(tmp)
        model = tf.nn.relu(model)
        model = tf.layers.conv2d(model, filters=int(n_filters*compression),kernel_size=3, padding='SAME', activation = None)

        if dropout>0:
            model = tf.nn.dropout(model, dropout)
        
        tmp = tf.concat([tmp,model],axis=-1)         
    return tmp

def TransitionBlock(model, n_filters = 12, dropout = 0.2, compression = 0.5):
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    model = tf.layers.conv2d(model, filters=int(n_filters*compression),kernel_size=1, padding='SAME', activation = None)

    if dropout>0:
        model = tf.nn.dropout(model, dropout)
    
    model = tf.layers.AveragePooling2D(pool_size = 2, strides=1)(model)
    return model

def DenseNet(Img,num_classes =10, n_filters = 36, dropout = 0.2, l= 6,compression = 0.5):
    
    model = tf.layers.conv2d(inputs = Img, filters=36,kernel_size=3, padding='SAME', activation = None)

    # Dense + Transition Block : 1
    model = DenseBlock(model, n_filters, dropout,l=3)
    model = TransitionBlock(model, n_filters, dropout)

    # Dense + Transition Block : 2
#     model = DenseBlock(model, n_filters, dropout,l)
#     model = TransitionBlock(model, n_filters, dropout)

#     # Dense + Transition Block : 3
#     model = DenseBlock(model, n_filters, dropout)
#     model = TransitionBlock(model, n_filters, dropout)

    # Dense + Transition Block : 4
    model = DenseBlock(model, n_filters, dropout,l=3)

    # output stage
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    model = tf.layers.AveragePooling2D(pool_size = 2, strides=1)(model)
    model = tf.layers.flatten(model)
    
    prLogits = tf.layers.dense(inputs = model,units=10, activation=None)
    
    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)

    return prLogits, prSoftMax
##############################################################################################

def bottleNeckUnit(model): 
    model = tf.layers.conv2d(model, filters=32,kernel_size=1, padding='SAME', activation = None)
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    
    model = tf.layers.conv2d(model, filters=64,kernel_size=3, padding='SAME', activation = None)
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)
    
    model = tf.layers.conv2d(model, filters=32,kernel_size=1, padding='SAME', activation = None)
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    return model

def merge(C_splits):
    model = C_splits[0]
    for i in range(1, len(C_splits)):
        model = tf.math.add(model,C_splits[i])
    return model

def splitUnit(model,cardinality=5):
    C_splits = []
    for i in range(cardinality) :
        split = bottleNeckUnit(model)
        C_splits.append(split)
        
    return C_splits

def ResBlock(model,cardinality=5):
    # split + transform(bottleneck) + transition + merge
    I =  model
    C_splits = splitUnit(model)
    model = merge(C_splits)
    model = tf.add(model,I)
    return model 

def ResNext(Img, num_classes=10, cardinality=5):

    # first layers
    model = tf.layers.conv2d(inputs = Img, filters=32,kernel_size=3, padding='SAME', activation = None)
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    model = ResBlock(model,cardinality=5)
    model = ResBlock(model,cardinality=5)

    model = tf.layers.flatten(model)
    model = tf.layers.dense(inputs = model,units=256, activation=None)
    prLogits = tf.layers.dense(model,units=10, activation=None)

    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)

    return prLogits, prSoftMax
