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


def read_data(filepath):
    
    colnames = []
    
    with open(filepath, 'r') as file:
        
        last_line = ""
        
        for line_number, line in enumerate(file, start=1):
            if line.startswith("%"):
                last_line = line
            else:
                colnames = last_line.replace('%',"").replace("\n","").split(";")
                break

    data = pd.read_csv(filepath, comment='%', sep=';',
                       names=colnames).values

    t = data[0:, 0]
    ch1 = data[0:, 1]
    ch2 = np.zeros(len(t))
    
    
    if len(colnames) > 2:
        if not np.isnan(data[0, 2]):
            ch2 = data[0:, 2]

    return t, ch1, ch2, colnames


def evalutae_noise(t, y, labels):
    fft_result = np.abs(np.fft.rfft(y))
    dt = t[1] - t[0]
    frequencies = np.fft.rfftfreq(len(y), d=dt)

    y = y - np.average(y)
    yrms =  np.sqrt(np.average(y**2))


    plt.plot(t, y)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    
    plt.hlines(yrms, t[0], t[-1], color="red")
    
    plt.title(str(round(1 / dt / 1E6, 1)) + "MS/s")
    plt.show()
    
    plt.plot(frequencies, np.abs(fft_result)**2)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    
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
            t, ch1, ch2, colnames = read_data(folder + '\\' + file)
            dt = t[1] - t[0]
            
            evalutae_noise(t, ch1, colnames[:2])
            
            if len(colnames) == 3:
                evalutae_noise(t, ch2, [colnames[0], colnames[2]])
            
               
        except UnicodeDecodeError as ude:
            print(ude)
            continue
        

#folder_noiseA = r"Z:\Group Members\Hecker, Marcel\240125_Measurement\Current\Noise MFIA"
#folder_noiseV = r"Z:\Group Members\Hecker, Marcel\240125_Measurement\Voltage\Noise MFIA"
folder_noiseVA = r"Z:\Group Members\Hecker, Marcel\240125_Measurement\Votlage & Current"

#evaluate_files(folder_noiseA)
#evaluate_files(folder_noiseV)

evaluate_files(folder_noiseVA)

    

