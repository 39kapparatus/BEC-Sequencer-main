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
import os
import json
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
        Image_GS= np.array(image['images'].tolist())[0]
        Image_GS = cv2.normalize(Image_GS, None, 0, 255, cv2.NORM_MINMAX)
        Image_GS = Image_GS.astype(np.uint8)
        '''image = image.convertToFormat(QImage.Format.Format_ARGB32)
        width = image.width()
        height = image.height()
        # Get pointer to the image data
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        # Convert to NumPy array (height, width, 4)
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
        # Convert from ARGB (Qt) to BGR (OpenCV) and then to grayscale
        Image_RGB = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        Image_GS = cv2.cvtColor(Image_RGB, cv2.COLOR_BGR2GRAY)'''
   
    ## The two lines below can be used to manually select a ROI.
    if select_ROI:
        ## Select a ROI and Crop the Image
        ROI = cv2.selectROI("Select the ROI",Image_GS)
        cv2.destroyAllWindows()
    
    X = ROI[1]
    Y = ROI[0]
    DX = ROI[3]
    DY = ROI[2]
    #Image_RGB = Image_RGB[X:(X+DX),Y:(Y+DY)]
    Image_GS = Image_GS[X:(X+DX),Y:(Y+DY)]
    
    ## Check if the ROI encloses the MOT in all Images
    
    #plt.imshow(Image_GS)
    #plt.show()

    CameraCount = np.sum(Image_GS)
    print(CameraCount)
    #Get the count
    Atom_Number = 4 * np.pi * Power(CameraCount, ExposureTime, Gain) / (h * v * Omega * LambdaPrime) * 1
    return Atom_Number,ROI


def extract_trap_aom_voltage(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Find Trap AOM FM voltage in the JSON
    # You mentioned the sweep_values list contains event with channel_name "Trap AOM FM"
    # Let's search for that:
    for event in data.get('sweep_values', []):
        if event.get('channel_name') == 'Trap AOM FM':
            return event.get('value')
    # If not found in sweep_values, fallback to sweep_dict (first entry)
    try:
        voltages = data["sweep_dict"]["Intialize Parameters|Trap AOM FM|target_value"]
        if voltages:
            return voltages[0]  # fallback: first voltage value
    except:
        pass
    return None

if __name__=='__main__':

    # Replace with the path to your folder
    folder_path = r'C:\Users\E3\Desktop\BEC-Sequencer-main (1)\BEC-Sequencer-main\Trigger_match\images'
    json_folder = r'C:\Users\E3\Desktop\BEC-Sequencer-main (1)\BEC-Sequencer-main\Trigger_match'

    # Filter only files (exclude subdirectories)
    files = os.listdir(folder_path)

    atom_numbers = []
    voltages = []

    for i, file in enumerate(files):
        print(file)
        image_data = np.load(os.path.join(folder_path, file), allow_pickle=True)
        ROI = [675, 511, 155, 133]
        if i == 0:
            out = Get_Atom_Number(ExposureTime=8 * 10 ** (-3), image=image_data, ROI = ROI)
            # ROI = out[1]
            # print(ROI)
        else:
            out = Get_Atom_Number(ExposureTime=8 * 10 ** (-3), image=image_data, ROI=ROI)

        atom_numbers.append(out[0])

        # Extract image grayscale and crop to ROI for display:
        Image_GS = np.array(image_data['images'].tolist())[0]
        Image_GS = cv2.normalize(Image_GS, None, 0, 255, cv2.NORM_MINMAX)
        Image_GS = Image_GS.astype(np.uint8)
        X, Y, DX, DY = ROI[1], ROI[0], ROI[3], ROI[2]
        cropped_img = Image_GS[X:(X + DX), Y:(Y + DY)]

        # # Display the cropped image with matplotlib
        plt.figure()
        plt.title(f"Image {i} - ROI cropped")
        #plt.imshow(cropped_img, cmap='gray',vmin=0, vmax=255)
        plt.imshow(cropped_img, cmap='gray')
        plt.axis('off')
        plt.show()

    # # Plot atom numbers over images
    # plt.figure()
    # plt.plot(voltages, atom_numbers)
    # plt.title('Atom Number vs Image Index')
    # plt.xlabel('Image Index')
    # plt.ylabel('Atom Number')
    # plt.show()
    voltages_np = np.array(voltages)
    atom_numbers_np = np.array(atom_numbers)

    # Filter out NaNs (if any)
    valid_indices = ~np.isnan(voltages_np)
    voltages_clean = voltages_np[valid_indices]
    atom_numbers_clean = atom_numbers_np[valid_indices]

    # Sort by voltage
    sorted_indices = np.argsort(voltages_clean)
    voltages_sorted = voltages_clean[sorted_indices]
    atom_numbers_sorted = atom_numbers_clean[sorted_indices]

    # Plotting
    plt.figure(figsize=(8,6))
    plt.plot(voltages_sorted, atom_numbers_sorted, 'o-', color='navy', markersize=6, linewidth=2, label='Atom Number')

    plt.title('Atom Number vs Trap AOM Voltage', fontsize=16)
    plt.xlabel('Trap AOM Voltage (V)', fontsize=14)
    plt.ylabel('Atom Number', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()