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
from PyQt5.QtGui import QImage

# Camera Calibration

# The Updated Camer Calibration at Zero Gain
def Power(CameraCount, ExposureTime, Gain):
    return CameraCount * (0.8 * 10 ** (-6))/2425133 * (113 * 10 ** (-6))/ExposureTime

#Q_Eff = 0.35 #the quantum efficiency is accounted for in the camera calibration

def Get_Atom_Number(ExposureTime= 2.391 * 10 ** (-3), ROI=[575,425,450,300],
                    select_ROI=False,
                    file_name=r'C:\Users\E3\Desktop\fluorence count\2025-05-06\test'+ str(2.391) + '.jpg',
                    image=None):
    #ROI is an array with 4 integers which delimit one corner of the ROI, the width and the height, ROI=[Y,X,DY,DX]

    Gain = 0

    # Converting to Number of Trapped Atoms

    Lambda = 1.84 * np.pi * 6.035 * 10 ** 6 # in Hz
    I_Sat = 1.75 * 10 ** (-3) # in W/cm2
    Delta = 2.5 * Lambda # in Hz

    # MOT Beams
    I_MOT = (300e-3) / (1.25**2) # in W/cm2

    # Power in each Photon
    h = 6.626 * 10 ** (-34) # in J/Hz
    v = 391 * 10 ** (12) # in Hz

    Omega = 0.5**2/12**2 # Solid Angle of the Camera

    LambdaPrime = Lambda/2 * (I_MOT / I_Sat) / (1 + (I_MOT/I_Sat) + (2*Delta/Lambda)**2)

    
    ## Upload the RGB Image and Convert it to Grayscale
    if image is None:
        #read the filename if image was not provided in the function call
        Image_RGB = cv2.imread(file_name)
        Image_GS = cv2.cvtColor(Image_RGB, cv2.COLOR_BGR2GRAY)
    else:
        image = image.convertToFormat(QImage.Format.Format_ARGB32)
        width = image.width()
        height = image.height()
        # Get pointer to the image data
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        # Convert to NumPy array (height, width, 4)
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
        # Convert from ARGB (Qt) to BGR (OpenCV) and then to grayscale
        Image_RGB = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        Image_GS = cv2.cvtColor(Image_RGB, cv2.COLOR_BGR2GRAY)
   
    ## The two lines below can be used to manually select a ROI.
    if select_ROI:
        ## Select a ROI and Crop the Image
        ROI = cv2.selectROI("Select the ROI",Image_GS)
        cv2.destroyAllWindows()
    
    X = ROI[1]
    Y = ROI[0]
    DX = ROI[3]
    DY = ROI[2]
    Image_RGB = Image_RGB[X:(X+DX),Y:(Y+DY)]
    Image_GS = Image_GS[X:(X+DX),Y:(Y+DY)]
    
    ## Check if the ROI encloses the MOT in all Images
    
    #plt.imshow(Image_GS)
    # plt.show()

    CameraCount = np.sum(Image_GS)
    #Get the count
    Atom_Number = 4 * np.pi * Power(CameraCount, ExposureTime, Gain) / (h * v * Omega * LambdaPrime) * 1
    return Atom_Number,ROI

if __name__=='__main__':
    print(Get_Atom_Number())