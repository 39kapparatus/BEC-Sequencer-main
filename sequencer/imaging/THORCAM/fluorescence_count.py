# -*- coding: utf-8 -*-
"""
Created on Tu May 5th 2025

@author: mathias
"""

# Basic Python Library
import numpy as np
from math import *
import cv2 # for Images
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Camera Calibration

# The Updated Camer Calibration at Zero Gain
def Power(CameraCount, ExposureTime, Gain):
    return CameraCount * (0.8 * 10 ** (-6))/2425133 * (113 * 10 ** (-6))/ExposureTime

ExposureTime = 311 * 10 ** (-6)
Gain = 0

# Converting to Number of Trapped Atoms

Lambda = 1.84 * np.pi * 6.035 * 10 ** 6 # in Hz
I_Sat = 1.75 * 10 ** (-3) # in W/cm2
Delta = 2.5 * Lambda # in Hz

# MOT Beams
I_MOT = 240*10**(-3)/2**2 # in W/cm2

# Power in each Photon
h = 6.626 * 10 ** (-34) # in J/Hz
v = 391 * 10 ** (12) # in Hz

Omega = 0.5**2/12**2 # Solid Angle of the Camera

LambdaPrime = Lambda/2 * (I_MOT / I_Sat) / (1 + (I_MOT/I_Sat) + (2*Delta/Lambda)**2)

Q_Eff = 0.35

def N_Atoms(CameraCount, ExposureTime, Gain):
    return 4 * np.pi * Power(CameraCount, ExposureTime, Gain) / (h * v * Omega * LambdaPrime) * 1 / Q_Eff

# Loading the Images with Coils Turned On

# Time_Stamp = np.arange(0.5,15.5,0.5)
Number_of_Images = 200

# Atom_Number = np.zeros(len(Time_Stamp))
Atom_Number = np.zeros(Number_of_Images)

for i in np.arange(0,Number_of_Images,1):
    
    ## Upload the RGB Image and Convert it to Grayscale
    
    file_name = r'C:\Users\E3\Desktop\Thor Cam Videos\2023_09_14\3D MOT 3A dispenser 14V cell heating\Load (' + str(i+1) + ').jpg'
    # file_name = r"C:\Users\mmohi\OneDrive\Desktop\Load Atoms Trial 5\Load (" + str(i+1) + ").jpg"
    Image_RGB = cv2.imread(file_name)
    Image_GS = cv2.cvtColor(Image_RGB, cv2.COLOR_BGR2GRAY)
    # C:\Users\E3\Desktop\Thor Cam Videos\2023_09_14\3D MOT 3A dispenser 14V cell heating
    ## The two lines below can be used to manually select a ROI.
    # ROI = cv2.selectROI("Select the ROI",Image_GS)
    # print(ROI)
    
    ## Select a ROI and Crop the Image
    
    X = 250
    Y = 510
    DX = 300
    DY = 450
    Image_RGB = Image_RGB[X:X+DX,Y:Y+DY]
    Image_GS = Image_GS[X:X+DX,Y:Y+DY]
    
    ## Check if the ROI encloses the MOT in all Images
    
    plt.imshow(Image_GS)
    # plt.show()
    
    # cv2.imshow('Original Image',Image_RGB)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    CameraCount = np.sum(Image_GS)
    Atom_Number[i] = N_Atoms(CameraCount, ExposureTime, Gain)

plt.figure(figsize=(6,6))
# plt.scatter(Time_Stamp, Atom_Number, s = 8)
plt.scatter(np.arange(0,Number_of_Images,1)/10, Atom_Number, s = 8)
plt.xlabel('\nTime [in Seconds]', fontsize = 14, weight = 'bold')
plt.ylabel('MOT Atom Number', fontsize = 14, weight = 'bold')
plt.title('MOT Atom Number', fontsize = 14, weight = 'bold')
#plt.savefig(r'C:\Users\E3\Desktop\Thor Cam Videos\2023_09_14\3D MOT 3A dispenser 14V cell heating')

plt.show()