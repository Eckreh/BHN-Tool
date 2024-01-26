# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:32:04 2024

@author: av179
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import powerlaw

folder = r"Z:\Group Members\Hecker, Marcel\240124_Measurement_Amplifier\MFIA 20x 250mV"

def sine(t, A, omega, phi):
    return A * np.sin(omega * t + phi)

def read_data(filepath):

    colnamesmfli = ['Time [s]', 'Ch1 [V]', 'Ch2 [V]']
    data = pd.read_csv(filepath, comment='%', sep=';',
                       names=colnamesmfli).values

    t = data[0:, 0]
    ch1 = data[0:, 1]
    ch2 = np.zeros(len(t))

    if not np.isnan(data[0, 2]):
        ch2 = data[0:, 2]

    return t, ch1, ch2


freqs = []
gain = []
phase = []

for file in os.listdir(folder):
    filename, ext = os.path.splitext(file)

    # Skip all files that arent txt.
    if not ext == ".txt":
        continue

    try:
        t, ch1, ch2 = read_data(folder + '\\' + file)
        dt = t[1] - t[0]

      
        tlin = np.linspace(t[0], t[-1], 100000)
        
        
        
        fft_result = np.abs(np.fft.rfft(ch2))
        frequencies = np.fft.rfftfreq(len(ch2), d=dt)
        
        peaks, _ = scipy.signal.find_peaks(fft_result, prominence=1, height=np.max(fft_result)/3)
        
       
        
        omega0 = (2*np.pi*frequencies[peaks[0]])
        
        popt1, pcov1 = scipy.optimize.curve_fit(sine, t, ch1, p0=[np.max(ch1), max(omega0, 6.3), 0], maxfev=100000000)
        popt2, pcov2 = scipy.optimize.curve_fit(sine, t, ch2, p0=[np.max(ch2), max(omega0, 6.3), 0], maxfev=100000000)
        
        freqs.append(popt2[1])
        gain.append(popt2[0]/popt1[0])
        phase.append(popt1[2]-popt2[2])
        
        
        plt.plot(t, ch1)
        plt.plot(t, ch2)
        
        plt.plot(tlin, sine(tlin, *popt1))
        plt.plot(tlin, sine(tlin, *popt2))
        
        plt.xlim(-(1/popt1[1] * 5),(1/popt1[1] * 5))
        plt.show()
        print(file)
        print(popt2[2]-popt1[2])
        
        
        
        #plt.plot(frequencies, fft_result) 
        #plt.plot(frequencies[peaks], fft_result[peaks], "x")
        #plt.show()
        
    except UnicodeDecodeError as ude:
        print(ude)
        continue
    

phase = np.asarray(phase)
 
plt.errorbar(freqs, 20*np.log10(gain), fmt='x')
plt.xscale('log')
plt.ylabel("Gain [dB]")
plt.xlabel("Frequency [Hz]")
plt.ylim((0,30))
plt.show()

phase = np.where(phase<0 , 2*np.pi+phase, phase)

plt.errorbar(freqs, np.degrees(phase), fmt='x')
plt.xscale('log')
plt.ylabel("Phase [°]")
plt.xlabel("Frequency [Hz]")

