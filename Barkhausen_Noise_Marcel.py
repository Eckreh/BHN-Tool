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
import helper_functions as hp
import sys


# %% Parameters
# Datafolder
folder = r"Z:\Group Members\Hecker, Marcel\240205\B3_9 bs selected diff Voltage (no new data)"

#folder = r"Z:\Group Members\Seiler, Toni\archive\measurement data shown in thesis\the ramped measurements\measurements that did show events\29.3kHz"

resistance = 50 #Ohm
threshold = 1E-5
maxlist=[]

smooth_data = False
 

# %% Functions

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

def lin_fit(x, a, b):
    return a*x + b

def analyze(data, xmin, xmax, binnumber, datatype='eventsize', folder ='.\\', ax = None):
    #params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
    #plt.rcParams.update(params)
    
    #plt.figure(figsize=(10,6))
    edges, hist = powerlaw.pdf(data, number_of_bins = binnumber)

    bin_centers = (edges[1:] + edges[:-1]) / 2.0

    fit=powerlaw.Fit(data)
    
    fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha,2), round(fit.sigma,2)), ax = ax)
    
    plt.scatter(bin_centers, hist)
    
    
# TODO: Check this code sometimes this results in empty data, othertime NaNs and infs
#->    
    hist = hist[xmin < bin_centers]
    bin_centers = bin_centers[xmin < bin_centers]

    hist = hist[bin_centers < xmax]
    bin_centers = bin_centers[bin_centers < xmax]

    bin_centers = bin_centers[hist != 0]
    hist = hist[hist != 0]

# <-

    if len(bin_centers) > 0 and len(hist) > 0:
        popt, pcov = scipy.optimize.curve_fit(lin_fit, np.log(bin_centers), np.log(hist))
        fitlin = np.exp(lin_fit(np.log(bin_centers), *popt))

        plt.plot(bin_centers, fitlin, label=r'$\alpha_{Lin}$'+r'={} $\pm${}'.format(np.abs(round(float(popt[0]),2)), round(float(np.sqrt(pcov[0][0])),2)))
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
    plt.savefig(folder + "\\powerlaw.png")
    plt.show()
    startvalue=xmin

    return fit, popt, pcov, bin_centers, hist
        
def analyzeBH(data_t, data_current, threshold):
    
    
    retlist = []
    
    jerksandbase = hp.calculate_derivative(data_t, data_current, "simple")**2
    
    baseline = hp.interpolate_baseline(data_t, jerksandbase)

    jerks=jerksandbase-baseline
    
    #plt.plot(data_t, jerksandbase)
    #plt.plot(data_t, baseline)
    #plt.show()
    
    
    filtered = np.copy(jerks)
    filtered[filtered < threshold] = 0 
    
    #plt.plot(data_t, jerks)
    #plt.plot(data_t, filtered)
    #plt.show()
    
#         More Copy from Toni    
#           identify the jerks where they breach the threshold and calculate the corresponding values that are to be analyzed
#           the indizes within the cutlist are saved
    
    durationlist=[]
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
                retlist.append(max(np.take(filtered, indexlist)))
                #reset for next jerk
                indexlist=[]
                wait=0
                duration=0
                
        else:
            indexlist.append(j-1)
            duration=duration+1
            indexlist.append(j)
    
    
    return retlist

# %% Evaluation

if __name__ == "__main__":
        
    plt.rc('figure', figsize=(16.0, 10.0))
    
    for file in os.listdir(folder):
        filename, ext=os.path.splitext(file)  
        
        #Skip all files that arent txt.
        if not ext == ".txt":
            continue
                
        try:    
               
            t, ch1, ch2, colnames = hp.read_data(folder + '\\' + file)
        
            if "(V)" in colnames[1]:
                current = ch1 / resistance
            elif "(A)" in colnames[1]:
                current = ch1
            else:
                print("THIS Broken")
            
            dt = t[1] - t[0]
    
        except UnicodeDecodeError as ude:
            print(ude)
            continue
        
        
        if len(colnames) == 3 and True:
            
            st = hp.SchmittTrigger(0.007, 0.001)
          
            trigger = [st.process_input(abs(x)) for x in ch2]
            trigger2 = hp.preplace_ones(trigger, hp.time_to_samples(20, "us", 1/dt))
            
            t_slice = hp.slice_array(t, trigger2)
            ch1_slice = hp.slice_array(ch1, trigger2)
            
    
            #TODO: maybe get more than 1 dataset from file with trigger if possible
            # have to find a way to detected if it is complete and not cut of or a non-switching peak (DWM, PUND)
            
            prob_index = hp.closest_to_zero_index(t_slice)
            
            #plt.plot(t, ch2)
            #plt.plot(t_slice[prob_index], ch1_slice[prob_index])
            t = np.asarray(t_slice[prob_index])
            current = np.asarray(ch1_slice[prob_index])
    
            # TODO: Probably not needed remove maybe
    
            ch1ds = scipy.signal.decimate(ch1_slice[prob_index], 2)
            ch1av = hp.combine_points(ch1_slice[prob_index])
            
              
        # Smooth Data
        if smooth_data:
            sos = scipy.signal.butter(4, 15E3, 'lowpass', fs=1/dt, output='sos')
            smooth_current = scipy.signal.sosfiltfilt(sos, current)
            left_idx, right_idx, _ = calculate_peak_width(t, smooth_current, 0.65)
        else:
            left_idx, right_idx, _ = calculate_peak_width(t, current, 0.65)
        
            
        # Plot Measured Data
        if True:
            plt.ylabel("Current [A]")
            plt.plot(t, current, label="Measurment")
            plt.xlabel("Time [s]")
            #left_idx += time_to_samples(0.2, "ms", 1/dt)
            plt.errorbar([t[left_idx], t[right_idx]], [current[left_idx], current[right_idx]], fmt='x')
            
            if smooth_data:
                plt.plot(t, smooth_current, label="LP-Filtered 15kHz")
            
            plt.legend()
            plt.savefig(folder + "\\data_" + filename.replace('.txt', '.png'))
            
            plt.show()
            
            plt.plot(t[left_idx:right_idx],current[left_idx:right_idx])
            
            if smooth_data:
                plt.plot(t[left_idx:right_idx],smooth_current[left_idx:right_idx])
                
            plt.savefig(folder+"\\peak_" + filename.replace('.txt','.png'))
            plt.show()
            
            
        # Plot (d/dt)
        # TODO: check if only peak is used else plot whole derivative
        if True:
            fig, ax1 = plt.subplots()
            plt.plot(t, hp.calculate_derivative(t, current, "simple"))
            
            plt.xlim(t[left_idx], t[right_idx])
            
            if smooth_data:
                ax2 = ax1.twinx()  
                ax2.set_ylim(ax1.get_ylim()[0]/10,ax1.get_ylim()[1]/10)
                
                plt.plot(t, hp.calculate_derivative(t, smooth_current, "simple"), color="red")
                
            plt.savefig(folder+"\\deriv_" + filename.replace('.txt','.png'))
            plt.show()
            
         # Plot FFT
        if False:
            fft_result = np.fft.rfft(current)
            frequencies = np.fft.rfftfreq(len(current), d=dt)
         
            plt.plot(frequencies, 10*np.log10(np.abs(fft_result)**2))
            plt.xscale('log')
            plt.show()
            
            
        # Use Only Peak in Analysis
        if False:
            t_section = t[left_idx:right_idx]
            current_section = current[left_idx:right_idx]
                
            current = current_section
            t = t_section
            
        
        maxlist.extend(analyzeBH(t, current, threshold))
    
    
    analyze(maxlist, xmin=6e-3, xmax=1E2, binnumber=50, folder=folder)
        
    
    print("DONE!")
   