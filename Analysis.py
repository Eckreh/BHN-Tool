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


## Adjust these: 
Vdiv = 41  # 41 @ 1MOhm or 23 @ 10MOhm
pre_addtime = 2E-4    # 2E-4 120Hz PD
pre_cuttime = 5E-4    # 4E-4 120Hz PD
trig_high = 0.01
trig_low = 0.005


clockbase = 60E6
pressed_key = -1


def on_press(event):
    global pressed_key
    pressed_key = event.key
    

def evaluate_section(t, ch1, ch2, dt, threasholds, filename, num=-1):
    global pressed_key
    global pre_addtime
    global pre_cuttime
    global trig_high
    global trig_low
    
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

    st = hf.SchmittTrigger(trig_high, trig_low)
    
    triggerUp = np.asarray([st.process_input(x) for x in ch2])
    st.reset()
    triggerDown = -1*np.asarray([st.process_input(-x) for x in ch2])
    st.reset()
    
    triggersequence, triggerindex = hf.extract_consecutive(triggerDown + triggerUp)

    fsm_waveformdetect.process_input(triggersequence)
    waveform = str(fsm_waveformdetect)
    
    fsm_waveformdetect.reset()
    trigger = [st.process_input(abs(x)) for x in ch2]
    
    
    triggermod = hf.preplace_ones(trigger, hf.time_to_samples(pre_addtime, "s", 1/dt))
    triggermod = hf.reduce_ones(triggermod, hf.time_to_samples(pre_cuttime, "s", 1/dt))
    
    sliced_ch1 = hf.slice_array(ch1, triggermod)
    sliced_ch2 = hf.slice_array(ch2, triggermod)
    sliced_t = hf.slice_array(t, triggermod)
    
    slice_test = hf.slice_array([1]*len(trigger), trigger)
    
    fsm_peaks = hf.StateMachine("Default",transitions_splp)
    
    if "PN" in waveform:
        
        edgets = t[triggerindex[np.nonzero(triggersequence)]]
        # TODO: Change this similar to DWM for captured fractions
        edgedts = hf.calculate_differences_every_nth(edgets,1)
        freq = 1/(2*np.average(edgedts))
        cond = abs(ch2) > np.max(abs(ch2))*0.96
        appV = np.average(abs(ch2[cond]))
        peaksequnce = [fsm_peaks.process_input([x]) for x in triggersequence[np.nonzero(triggersequence)]]
        
        
    elif "DWM" in waveform:
        #TODO: this doesnt seem to work @ all Samplerates
        peaks, _ = scipy.signal.find_peaks(np.abs(np.fft.rfft(ch2)), prominence=150)
        freqs = np.fft.rfftfreq(ch2.size, d=dt)
        
        #TODO: Check len peaks
        
        if len(peaks) > 3:
            idx1 = 2
            idx2 = 3
        else: # 0.6 fraction and other maybe larger needs to be checked
            idx1 = 1
            idx2 = 2
        
        if freqs[peaks][idx1]/freqs[peaks][idx2] > 0.6:
            waveform += "_TR"
            appV = np.max(abs(ch2))
        else:
            waveform += "_SQ"
            cond = abs(ch2) > np.max(abs(ch2))*0.96
            appV = np.average(abs(ch2[cond]))
        
        edgets = np.asarray(t[triggerindex[np.nonzero(triggersequence)]])
        edgedts = np.asarray(hf.calculate_differences_every_nth(edgets, 1))
        
        edgedts = hf.filter_array(edgedts, 20)
        freq = 1/(4*np.mean(edgedts))
        peaksequnce = [fsm_peaks.process_input([x]) for x in triggersequence[np.nonzero(triggersequence)]]
        
    else:
        print("Waveform couldnt be Detected!")
        return [],[]
        
        
    ## TODO: if avg_len changes in file this wont adopt
    avg_len = -1

    if avg_len < 0:
        for bt in sliced_t:
            avg_len += len(bt)
            
        avg_len /= len(sliced_t)
    
    
    
    if len(peaksequnce) != len(sliced_t):
        testlen = 0
        
        for sl in slice_test:
            testlen += len(sl)
            
        testlen /= len(slice_test)
        
        if len(slice_test[0]) < 0.4*testlen:
            peaksequnce = peaksequnce[1:]
    
   
    
    for sch1, st, pt, sch2 in zip(sliced_ch1, sliced_t, peaksequnce, sliced_ch2):
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
                    
                    risetime = -1
                    
                    if "PN" in waveform or "DWM_SQ" in waveform:
                        absvals = np.abs(sch2)
                        maxval = np.max(absvals)
                        stnp = np.asarray(st)
                        risefall = stnp[(absvals > 0.1 * maxval) & (absvals < 0.9 * maxval)]
                        
                        if len(risefall) - (risefall[-1] - risefall[0])/dt > 2:
                            print("RISETIME ERROR!")
                            risetime = -2
                        else:
                            risetime = risefall[-1] - risefall[0]
                    else:
                        print("Doesnt Apply")
                    
                    
                    metadata = {
                        "Vapp": appV*Vdiv,
                        "Vraw": appV,
                        "threashold_spike": threashold_spike,
                        "threashold_continus": threashold_continus,
                        "samples_to_average": samples_to_average,
                        "waveform": waveform,
                        "risetime": risetime,
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
    
    #if len(filtered_peaks) > 0:
    #    np.save("good" + starttime, hf.pad_arrays_to_4096(good_peaks))
    
    #if len(discard_peaks) > 0:
    #    np.save("bad" +  starttime, hf.pad_arrays_to_4096(discard_peaks))
    
    
    return filtered_peaks, parameters


def evaluate_scope_file(data, threasholds, filename):
    global Vdiv
    global clockbase
   
    filetype = hf.determine_filetype(data)
    
    arr0 = [] 
    arr1 = []
    
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
    
            a0, a1 = evaluate_section(t, ch1, ch2, dt, threasholds, filename, num)
            
            if len(a1) > 0:
                arr0.extend(a0)
                arr1.extend(a1)
                
    elif filetype == "CSV":
        
        t, ch1, ch2, _ = data
        dt = t[1]-t[0]
        arr0, arr1 = evaluate_section(t, ch1, ch2, dt, threasholds, filename)
        
    
    return arr0, arr1



plt.ion()
plt.connect("key_press_event", on_press)

folder = r"D:\Session 240226\BTA C10 mb 3_4 363K\DWM Things\pass"


ar1 = []
ar2 = []

thresholdsPVDF = {20: [2, 2E-10],
                  29: [2, 6.1E-10],
                  33: [6, 10.29E-10]}
  
thresholdsBTAC10 = {16: [0.1, 0.30]}

thresholdsNone = {0: [0, 0]}


if not os.path.exists(os.path.join(folder,"Used")):
    os.makedirs(os.path.join(folder, "Used"))
     
if not os.path.exists(os.path.join(folder,"NotUsed")):
    os.makedirs(os.path.join(folder, "NotUsed"))


for file in os.listdir(folder):
    
    filename, ext = os.path.splitext(file)
    
    if ext == ".npy":
        print("Eval'ing: " + str(filename))
        a1, a2 = evaluate_scope_file(np.load(folder + '\\' + file, allow_pickle = True), thresholdsBTAC10, str(filename))
        
    elif ext == ".txt":
        print("Eval'ing: " + str(filename))
        a1, a2 = evaluate_scope_file(hf.read_data(folder + '\\' + file), thresholdsBTAC10, str(filename))
    else:
        continue
                                     
    if len(a1) > 0:
        ar1.append(a1)
        ar2.append(a2)
        shutil.move(os.path.join(folder, file), os.path.join(folder, 'Used', file))
    else:
        shutil.move(os.path.join(folder, file), os.path.join(folder, 'NotUsed', file))
        


starttime = datetime.now().strftime("%Y-%m-%d_%H-%M")

flatar1 = [item for sublist in ar1 for item in sublist]
flatar2 = [item for sublist in ar2 for item in sublist]
npar1 = np.asarray(flatar1, dtype="object")

print("break here")
np.savez_compressed("scope_eval_" + str(starttime), npar1, flatar2)

print("exit")