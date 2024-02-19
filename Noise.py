# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:50:02 2024

@author: av179
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import powerlaw
import helper_functions as hp


def evalutae_noise(t, y, labels):
   
    dt = t[1] - t[0]
    y = y - np.average(y)
    
    #fft_result = np.abs(np.fft.rfft(y))
    #frequencies = np.fft.rfftfreq(len(y), d=dt)

    
    yrms =  np.sqrt(np.average(y**2))
    #frequencies, psd = plt.psd(y, Fs=1/dt)

    #plt.title('PSD: ' + labels[1])
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Power/Frequency (dB/Hz)')
    #plt.grid(True)
    #plt.show()

    plt.plot(t, y)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    
    plt.hlines(yrms, t[0], t[-1], color="red")
    
    plt.title(str(round(1 / dt / 1E6, 1)) + "MS/s")
    plt.show()
    
    #plt.plot(frequencies, np.abs(fft_result)**2)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
    
    
    if "(V)" in labels[1]:
        print(str(yrms) + " Vrms " + str(yrms/50) + " Arms @ 50Ohm")
    else:
        print(np.average(yrms), "Arms")
        
    
def evaluate_files(folder):
    for file in os.listdir(folder):
        filename, ext = os.path.splitext(file)

        # Skip all files that arent txt.
        if not ext == ".txt":
            continue

        try:
            t, ch1, ch2, colnames = hp.read_data(folder + '\\' + file)
            dt = t[1] - t[0]
            
            evalutae_noise(t, ch1, colnames[:2])
            
            if len(colnames) == 3:
                evalutae_noise(t, ch2, [colnames[0], colnames[2]])
            
               
        except UnicodeDecodeError as ude:
            print(ude)
            continue
        

#folder_noiseA = r"Z:\Group Members\Hecker, Marcel\240125_Measurement\Current\Noise MFIA"
#folder_noiseV = r"Z:\Group Members\Hecker, Marcel\240125_Measurement\Voltage\Noise MFIA"
folder_noiseVA = r"Z:\Group Members\Hecker, Marcel\240125_Measurement\Votlage & Current Noise 60MSs"

#evaluate_files(folder_noiseA)
#evaluate_files(folder_noiseV)

evaluate_files(folder_noiseVA)

    

