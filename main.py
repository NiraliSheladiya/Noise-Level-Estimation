# Author : Nirali Sheladiya

import cv2
import numpy as np
import sklearn.feature_extraction.image as skf
import scipy.stats as sp
import math

def generateToeplitz(derivativeMatrix, PS):
        numRow, numCol = np.shape(derivativeMatrix)
        Temp = np.zeros([(PS - numRow + 1)*(PS - numCol + 1) + 1, (PS * PS) + 1])
        T = np.zeros([(PS - numRow + 1) * (PS - numCol + 1), (PS * PS)])
        rowNum = 1
        for i in range(1, PS - numRow + 2):
            for j in range(1, PS - numCol + 2):
                for p in range(1, numRow + 1):
                    Temp[rowNum][((i - 1 + p - 1) * PS + (j - 1) + 1):((i - 1 + p - 1) * PS + (j - 1) + 2 + numCol - 1)]= derivativeMatrix[p - 1][:]
                rowNum += 1
        T = Temp[1:, 1:]
        return T


def computeVariance(patchCollection, patchSize):
    covOfImage = np.zeros([patchSize * patchSize, patchSize * patchSize])
    _, numP = np.shape(patchCollection)
    #print(numP)
    covOfImage=patchCollection.dot(patchCollection.T)/ (numP - 1)
    Sigma=np.around(covOfImage, decimals=2)
    eigenValue, vect = np.linalg.eig(Sigma)   
    varValue = np.min(eigenValue)
    varValue=np.around(varValue, decimals=2)
    #print("std dev :", math.sqrt(varValue))
    return varValue


def noiseLevelEstimation(noisyImage, patchSize=7, confidenceLevel=1-1e-6, numIteration=10):
    """
    Input:
        pathI : Path for noisy RGB image
        patchSize= Image patch size
        confidenceLevel:  select close to 1
        numIteration : Max number of iterations
    """
    noisyImage = np.around(noisyImage, decimals=2)
    if (patchSize < 3):
            print("Patch size must be greater than or equal to 3")
            return None
    # Horizantal and vertical derivative operators
    horizontalKernel = np.ones((1, 3), np.float32)
    horizontalKernel[0][0], horizontalKernel[0][1], horizontalKernel[0][2] = -1/2, 0, 1/2
    verticalKernel = np.ones((3, 1), np.float32)
    verticalKernel[0][0], verticalKernel[1][0], verticalKernel[2][0] = -1/2, 0, 1/2
    # Toeplitz form of derivative operators
    Dh = generateToeplitz(horizontalKernel, patchSize)
    Dv = generateToeplitz(verticalKernel, patchSize)
    DD = ((Dh.T).dot(Dh)) + ((Dv.T).dot(Dv))
    # Inverse gamma CDF computation for given confidence interval
    k1 = np.matrix.trace(DD)
    r = np.linalg.matrix_rank(DD)
    inverseGammaCDF = 2 * sp.gamma.ppf(confidenceLevel, a=((float(r)) / 2), loc=0, scale=k1 / float(r))
    inverseGammaCDF=np.around(inverseGammaCDF, decimals=2)
    thresold = np.zeros([numIteration, 1])
    var = np.zeros([numIteration, 1])
    # Low rank patch selection(Iterative Framework)
    estimatedVariance = 0
    for i in range(numIteration):
        if (i == 0):
            # variance computation and patch collection for first iteration
            patchesofImage = skf.extract_patches_2d(noisyImage, (patchSize, patchSize))
            MAX_noPatches, _, _ = np.shape(patchesofImage)
            Ipatches = np.zeros([patchSize * patchSize, MAX_noPatches], dtype=float)
            for n in range(MAX_noPatches):
                    a = patchesofImage[n, :, :]
                    aa = a.reshape(patchSize * patchSize, 1)
                    Ipatches[:, n] = aa[:, 0]
            patchCollection=Ipatches
                    
        var[i, 0] = computeVariance(patchCollection, patchSize)
        thresold[i, 0] = var[i, 0] * inverseGammaCDF
        thresold[i, 0]=np.around(thresold[i, 0], decimals=2) 
        _, numP = np.shape(patchCollection)
        tempCollection = np.zeros([patchSize * patchSize, numP])
        count = 0
        textureStrength = np.zeros([numP, 1])
        for n in range(numP):
            patch = patchCollection[:, n]
            grad = np.stack((Dh.dot(patch), Dv.dot(patch)), axis=1)
            cov = (grad.T).dot(grad)
            textureStrength[n,0]=np.matrix.trace(cov)           
            # eigenValue, vect = np.linalg.eig(cov)
            # textureStrength[n, 0] = np.sum(eigenValue)
            # Finding Low rank patches out of patch collection
            if (textureStrength[n, 0] < thresold[i, 0]):
                tempCollection[:, count] = patch
                count = count + 1  # count: number of Low rank patches
        estimatedVariance = var[i, 0]
        # Stoping criteria for stable varince        
        if (count < patchSize * patchSize):
            break
        if (var[i,0] < 0.1):
                print("Noise free image")
                break
        if (i > 0):
            if (abs(var[i, 0] - var[i - 1, 0]) <= 0.1):
                break
        #print("thresold , count : ",thresold[i],count)
        patchCollection = np.zeros([patchSize*patchSize, count])
        patchCollection[:, 0:] = tempCollection[:, 0:count]
    return estimatedVariance


# Select the path for image file
pathI="C:\\Users\\Nirali IITian\\Desktop\\IQA\\Noise estimation\\_noise\\dataset\\test_1.jpg"
image = cv2.imread(pathI)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

patchSize = 7
confidenceLevel = 1 - 1e-6  # choose close to 1
numIteration = 3

stdArray = np.zeros([26, 2])
for i in range(1,26,1):
    row, col, ch = image.shape
    # Generation of Noisy image
    std = i
    var = std * std
    mean = 0.0
    noisyImage = grayImage + std * np.random.randn(row,col) #np.random.normal(mean, std, grayImage.shape)    
    # Noise Estimation
    EV = noiseLevelEstimation(noisyImage, patchSize, confidenceLevel, numIteration)
    stdArray[i, 0] = i
    if (EV >0):
      stdArray[i, 1] = math.sqrt(EV)
      print("Added Noise std: {} , Estimated Noise std: {} ".format(std, math.sqrt(EV) ))


# Plotting Graph
import matplotlib.pyplot as plt
plt.plot(stdArray[:,0], stdArray[:,0], label = "Added value") 
plt.plot(stdArray[:,0], stdArray[:,1], label = "Estimated value") 
plt.xlabel("Added noise std")
plt.ylabel("Estimated noise std")
plt.legend()
plt.title('Noise Estimation')
plt.show()
