# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:44:32 2024

@author: av179
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import powerlaw



# %% Parameters
# Datafolder
folder = r'Z:\Group Members\Hecker, Marcel\240125_Measurement\Voltage\B3_1 PUND 100Hz 3.75M 2.9Vpp'
#folder = r"Z:\Group Members\Seiler, Toni\archive\measurement data shown in thesis\the ramped measurements\measurements that did show events\29.3kHz"

resistance = 50 #Ohm
threshold = 7E-6
maxlist=[]

smooth_data = True
 

# %% Functions
def read_data(filepath):
    
    colnamesmfli = ['Time [s]', 'Ch1 [V]', 'Ch2 [V]']
    data = pd.read_csv(filepath, comment='%', sep=';', names=colnamesmfli).values
    
    t = data[0:, 0]
    ch1 = data[0:, 1]
    ch2 = np.zeros(len(t))
    
    if not np.isnan(data[0, 2]):
        ch2 = data[0:, 2]

    return t, ch1, ch2


def calculate_peak_width(data_x, data_y, percentage):
    
    #peak_index = scipy.find_peaks(np.abs(data), height=max(np.abs(data)/2))[0]
    
    peak_index = np.argmax(data_y)
    
    height = data_y[peak_index]*percentage
    
    left_idx = np.argmin(np.abs(data_y[:peak_index] - height))
    right_idx = np.argmin(np.abs(data_y[peak_index:] - height)) + peak_index
    
    _, _, left, right = scipy.signal.peak_widths(data_y, [peak_index], percentage)
    
    width = data_x[right_idx] - data_x[left_idx]
    
    #left_width = data_x[peak_index] - data_x[left_idx]
    #right_width = data_x[right_idx] - data_x[peak_index]


### TODO: fix this garbage!
    return int(left), int(right), width
    #return left_idx, right_idx, width


def calculate_derivative(data_x, data_y, order="second"):
    
    
    if order == "first":
        
        # f'(x) = (f(x+h)-f(x))/h
        
        data_yp = np.zeros_like(data_y)
        
        
        for i, f in enumerate(data_y):
            if i > 0 and i < len(data_y)-1:
                dt = data_x[i+1] - data_x[i]
                data_yp[i] = (data_y[i+1] - f) / dt
        
        return data_yp
        
    elif order == "second":
        ### TODO: this looks broken!
        # f'(x) = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
        data_yp = np.zeros_like(data_y)
        
        for i, f in enumerate(data_y):
            if i > 0 and i < len(data_y)-1:
                dt = data_x[i+1]-data_x[i]
                data_yp[i] = (data_y[i+1] - 2*f + data_y[i-1]) / dt**2
        
        return data_yp


def time_to_samples(time, unit, samplerate):
    
    samplerate = int(samplerate)
    
    if unit == "s":
        return int(samplerate * time)
    elif unit == "ms":
        return int(samplerate * (time/1000))
    else:
        raise NotImplementedError() 
        return None    
  
    
def lin_fit(x, a, b):
    return a*x + b

def analyze(data, xmin, xmax, binnumber, datatype='eventsize'):
    params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
    plt.rcParams.update(params)
    
    #plt.figure(figsize=(10,6))
    edges, hist = powerlaw.pdf(data, number_of_bins=binnumber)

    bin_centers = (edges[1:] + edges[:-1]) /2.0

    fit=powerlaw.Fit(data)
    
    fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers, hist)
    
    
# TODO: Check this code sometimes this results in empty data, othertime NaNs and infs
#->    
    hist=hist[xmin<bin_centers]
    bin_centers=bin_centers[xmin<bin_centers]

    hist=hist[bin_centers<xmax]
    bin_centers=bin_centers[bin_centers<xmax]

    bin_centers=bin_centers[hist!=0]
    hist=hist[hist!=0]

# <-

    if len(bin_centers) > 0 and len(hist) > 0:
        popt, pcov = scipy.optimize.curve_fit(lin_fit, np.log(bin_centers), np.log(hist))
        fitlin = np.exp(lin_fit(np.log(bin_centers), *popt))
        plt.plot(bin_centers,fitlin,label=r'$\alpha_{Lin}$'+r'={} $\pm${}'.format(np.abs(round(float(popt[0]),2)),round(float(np.sqrt(pcov[0][0])),2)))
    else:
        fit = None
        popt = None
        pcov = None
        
    plt.xscale('log')
    plt.yscale('log')
    if datatype=='eventsize':
        plt.xlabel(r'Slew-Rate $S=\left[\frac{A^2}{s^2}\right]$',fontsize=18)
    if datatype=='eventenergy':
        plt.xlabel('integrated eventsize')
    if datatype=='intereventtime':
        plt.xlabel('interevent time [s]')
        
    plt.ylabel(r'probability P($S=S_i$)',fontsize=18)
    plt.legend(fontsize=18)
    # plt.grid(True)
    plt.show()
    startvalue=xmin

    return fit, popt, pcov, bin_centers, hist
        


# %% Evaluation

plt.rc('figure', figsize=(16.0, 10.0))

for file in os.listdir(folder):
    filename, ext=os.path.splitext(file)  
    
    #Skip all files that arent txt.
    if not ext == ".txt":
        continue
            
    try:    
           
        t, ch1, ch2 = read_data(folder + '\\' + file)
        current = ch1 / resistance
        dt = t[1] - t[0]

    except UnicodeDecodeError as ude:
        print(ude)
        continue
    
     
    # Smooth Data
    if smooth_data:
        sos = scipy.signal.butter(2, 5E4, 'lowpass', fs=1/dt, output='sos')
        smooth_current = scipy.signal.sosfiltfilt(sos, current)
        left_idx, right_idx, _ = calculate_peak_width(t, smooth_current, 0.9)
    else:
        left_idx, right_idx, _ = calculate_peak_width(t, current, 0.9)
    
        
    # Plot Measured Data
    if True:
        plt.plot(t, current)
        #left_idx += time_to_samples(0.2, "ms", 1/dt)
        plt.errorbar([t[left_idx], t[right_idx]], [current[left_idx], current[right_idx]], fmt='x')
        
        if smooth_data:
            plt.plot(t, smooth_current)
            
        plt.show()
        
    
    # Plot (d/dt)
    if True:
        fig, ax1 = plt.subplots()
        plt.plot(t, calculate_derivative(t, current, "first"))
        #plt.plot(t, calculate_derivative(t, current, "second"))
        
        if smooth_data:
            ax2 = ax1.twinx()  
            ax2.set_ylim(ax1.get_ylim()[0]/10,ax1.get_ylim()[1]/10)
            
            plt.plot(t, calculate_derivative(t, smooth_current, "first"), color="red")
            
        
        plt.show()
        
     # Plot FFT
    if False:
        fft_result = np.fft.rfft(current)
        frequencies = np.fft.rfftfreq(len(current), d=dt)
     
        plt.plot(frequencies, 10*np.log10(np.abs(fft_result)**2))
        plt.xscale('log')
        plt.show()
        
        
    # Use Only Peak in Analysis
    if True:
        t_section = t[left_idx:right_idx]
        current_section = current[left_idx:right_idx]
            
        current = current_section
        t = t_section
        
    
# %% Copy from Toni
    
    Jerksandbase = calculate_derivative(t, current, "first")**2
    jerksminusfit=Jerksandbase
    
    #plt.plot(t, current, label="cut to peak 10%")
    
    minlist=[]
    timelist=[]
    
    #remove baseline made from local minima can be set to either only take negative minima for baseline or everything 
    for l in range(0, len(jerksminusfit)):
        
         if l == 0:
             if jerksminusfit[l]<jerksminusfit[l+1]:
                 minlist.append(jerksminusfit[l])
                 timelist.append(t[l])
     
         elif l == len(jerksminusfit)-1:
             if jerksminusfit[l]<jerksminusfit[l-1]:
                 minlist.append(jerksminusfit[l])
                 timelist.append(t[l])
     
         elif l > 0 and l < len(jerksminusfit)-1:
             if jerksminusfit[l] < jerksminusfit[l-1] and jerksminusfit[l] < jerksminusfit[l+1]:
                 minlist.append(jerksminusfit[l])
                 timelist.append(t[l])


    baseline = scipy.interpolate.pchip_interpolate(timelist, minlist, t)
    jerks=jerksminusfit-baseline
    
    filtered = np.copy(jerks)
    filtered[filtered < threshold] = 0
    
    
    #plt.plot(t, Jerksandbase)
    #plt.plot(timelist, minlist)
    #plt.plot(t, baseline)
    #plt.plot(t, jerks)
    #plt.plot(t, filtered, label="filtered, cut of: " + str(threshold))
    #plt.plot()
    #plt.legend()
    #plt.show()
    
    
    
# %%        More Copy from Toni    
#           identify the jerks where they breach the threshold and calculate the corresponding values that are to be analyzed
#           the indizes within the cutlist are saved
    
    durationlist=[]
    split = False
    varlist=[[],[]]
    sizelist=[] 
    waitlist=[]  
    indexlist=[]
    wait = 0
    duration = 0
    var = 0
    
    for j in range(0, len(filtered)):
        
        if filtered[j]==0:
            wait=wait+1
            
            if len(indexlist) != 0:
                # this is redundant since filtered[j] == 0 ?
                indexlist.append(j)
                #take the amplitude as maximum and put it into the list
                maxlist.append(max(np.take(filtered, indexlist)))
                #reset for next jerk
                indexlist=[]
                wait=0
                duration=0
                
        else:
            indexlist.append(j-1)
            duration=duration+1
            indexlist.append(j)
    

analyze(maxlist, xmin=2e0, xmax=30e2, binnumber=50)
    

print("Exit")
   

