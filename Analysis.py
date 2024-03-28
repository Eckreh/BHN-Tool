# -*- coding: utf-8 -*-

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import helper_functions as hf
import os
import scipy
import powerlaw
import sys
import shutil
from mpl_axes_aligner import align


Vdiv = 41  # or 23
clockbase = 60E6



def evaluate_section(t, ch1, ch2, dt, threasholds, filename, num=-1):

    filtered_peaks = []
    discard_peaks = []
    good_peaks = []
    parameters = []
    
    transitions_dwm_detect = {
        'Default':  {1: 'Up', -1: 'Down', 0: 'Default'},
        'Up' :      {1: 'DWM', -1: 'PN_-1', 0: 'Up'},
        'PN_-1' :    {1: "PN_1", -1: 'DWM', 0: 'PN_-1'},
        'Down':     {1: 'PN_1', -1: 'DWM', 0: 'Down'},
        'PN_1':     {1: 'DWM', -1: 'PN_-1', 0: 'PN_1'},
        'DWM':    {1: 'DWM', -1:'DWM', 0: 'DWM'},
    }
    
    
    transitions_splp = {
        'Default'   : {1: 'IS1', 0: 'Default', -1: 'IS-1'},
        'IS1'       : {1: 'LP1', 0: 'IS1', -1: 'SP-1'},
        'IS-1'      : {1: 'SP1', 0: 'IS1', -1: 'LP-1'},
        'LP1'       : {1: 'LP1', 0: 'LP1', -1: 'SP-1'},
        'LP-1'      : {1: 'SP1', 0: 'LP-1', -1: 'LP-1'},
        'SP1'       : {1: 'LP1', 0: 'SP1', -1: 'SP-1'},
        'SP-1'      : {1: 'SP1', 0: 'SP-1', -1: 'LP-1'}
        }
    
    fsm_waveformdetect = hf.StateMachine("Default", transitions_dwm_detect)


    # TODO: get frequency, risetime
    st = hf.SchmittTrigger(0.01, 0.005)
    
    triggerUp = np.asarray([st.process_input(x) for x in ch2])
    st.reset()
    triggerDown = -1*np.asarray([st.process_input(-x) for x in ch2])
    st.reset()
    
    triggersequence, triggerindex = hf.extract_consecutive(triggerDown + triggerUp)

    fsm_waveformdetect.process_input(triggersequence)
    waveform = str(fsm_waveformdetect)
    
    fsm_waveformdetect.reset()
    trigger = [st.process_input(abs(x)) for x in ch2]
    
    trigger = hf.preplace_ones(trigger, hf.time_to_samples(2E-4, "s", 1/dt))
    trigger = hf.reduce_ones(trigger, hf.time_to_samples(3E-4, "s", 1/dt))
    
    sliced_ch1 = hf.slice_array(ch1, trigger)
    sliced_t = hf.slice_array(t, trigger)
    
    if "PN" in waveform:
        
        edgets = t[triggerindex[np.nonzero(triggersequence)]]
        # TODO: Change this similar to DWM for captured fractions
        edgedts = hf.calculate_differences_every_second(edgets)
        freq = 1/np.average(edgedts)
        cond = abs(ch2) > np.max(abs(ch2))*0.96
        appV = np.average(abs(ch2[cond]))
        
    elif "DWM" in waveform:
        
        peaks, _ = scipy.signal.find_peaks(np.abs(np.fft.rfft(ch2)), prominence=200)
        
        if peaks[1]/peaks[2] < 2:
            waveform += "_TR"
            appV = np.max(abs(ch2))
        else:
            waveform += "_SQ"
        
        edgets = np.asarray(t[triggerindex[np.nonzero(triggersequence)]])
        edgedts = np.asarray(hf.calculate_differences_every_nth(edgets, 1))
        
        edgedts = hf.filter_array(edgedts, 20)
        freq = 1/(4*np.mean(edgedts))
        
        fsm_peaks = hf.StateMachine("Default",transitions_splp)
        
        peaksequnce = [fsm_peaks.process_input([x]) for x in triggersequence[np.nonzero(triggersequence)]]
        print(peaksequnce)
        
    else:
        raise NotImplemented
        
    
    ## TODO: if avg_len changes in file this wont adopt
    avg_len = -1

    if avg_len < 0:
        for bt in sliced_t:
            avg_len += len(bt)
            
        avg_len /= len(sliced_t)


    for sch1, st, pt in zip(sliced_ch1, sliced_t, peaksequnce):
        if len(st) > avg_len*0.7:
            
            deriv = hf.calculate_derivative(st, sch1)**2
            baseline = hf.interpolate_baseline(st, deriv)
            maxdb = np.max(deriv-baseline)
            
            samples_to_average = 16
            
            convol = np.convolve(sch1, np.ones(samples_to_average)/samples_to_average, mode='valid')
            diff = scipy.integrate.simpson(abs(sch1[:-(samples_to_average-1)]-convol),x=st[:-(samples_to_average-1)])
            
            threashold_spike, threashold_continus = threasholds[hf.closest_key(threasholds, appV*Vdiv)]
            
            if (maxdb > threashold_spike) or (diff > threashold_continus):

                plt.plot(st, sch1, label=str(maxdb) + " " + str(diff*1E10))
                plt.title(pt)
                plt.draw()
                
                while not plt.waitforbuttonpress():
                    print("looping")
                    
                plt.cla()
                
                if pressed_key == " ":
                    print("KEEP:" + str(maxdb) + " " + str(diff*1E10) + " @ " + str(appV*Vdiv))
                    filtered_peaks.append([st, sch1])
                    good_peaks.append([st-st[0], abs(np.asarray(sch1))])
                    
                    metadata = {
                        "Vapp": appV*Vdiv,
                        "Vraw": appV,
                        "threashold_spike": threashold_spike,
                        "threashold_continus": threashold_continus,
                        "samples_to_average": samples_to_average,
                        "waveform": waveform,
                        "peaktype": pt,
                        "frequency": freq,
                        "Samplerate": 1/dt,
                        "file": filename,
                        "num": num
                        }
                    
                    print(metadata)
                    
                    parameters.append(metadata)
                    
                else:
                    print("DISCARD:" + str(maxdb) + " " + str(diff*1E10) + " @ " + str(appV*Vdiv))
                    discard_peaks.append([st-st[0],abs(np.asarray(sch1))])
                    
                pressed_key = -1
                time.sleep(0.1)
                
            else:
                discard_peaks.append([st-st[0],abs(np.asarray(sch1))])
                    
    #plt.legend()
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if len(filtered_peaks) > 0:
        np.save("good" + starttime, hf.pad_arrays_to_4096(good_peaks))
    
    if len(discard_peaks) > 0:
        np.save("bad" +  starttime, hf.pad_arrays_to_4096(discard_peaks))
    
    
    return filtered_peaks, parameters


def evaluate_scope_file(data, threasholds, filename):
    global pressed_key
    global Vdiv
    global clockbase
   
    filetype = hf.determine_filetype(data)
    
    if filetype == "NPY":
        
        for key in data.item().keys():
            if "/dev" in key:
                wave_nodepath = key
    
        for num, record in enumerate(data.item()[wave_nodepath]):
            timestamp = record[0]["timestamp"]
            triggertimestamp = record[0]["triggertimestamp"]
            totalsamples = record[0]["totalsamples"]
            dt = record[0]["dt"]
    
            t = np.arange(-totalsamples, 0) * dt + (timestamp - triggertimestamp) / float(clockbase)
            ch1 = record[0]["wave"][0, :]
            ch2 = record[0]["wave"][1, :]
    
            evaluate_section(t, ch1, ch2, dt, threasholds, threasholds, num)


testdata = np.load(r"D:\Session 240209 B3_9 sb\raw\2024-02-09_15-11-36.npy", allow_pickle=True)
evaluate_scope_file(testdata,{0:[0,0]},"testfile")