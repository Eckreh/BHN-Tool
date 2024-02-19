# Shamelessly adapted from:
# https://github.com/zhinst/labone-api-examples/blob/release-24.1/common/python/example_scope_dig_stream.py
# https://github.com/zhinst/labone-api-examples/blob/release-24.1/common/python/example_scope_dig_dualchannel.py

# %% imports
import time
import warnings
import numpy as np
import zhinst.utils
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import helper_functions as hf
import os
import scipy
import Barkhausen_Noise_Marcel as bhm
import powerlaw
import sys


# %% functions

def get_scope_records(device, daq, scopeModule, num_records=1):
    """
    Obtain scope records from the device using an instance of the Scope Module.
    """

    # Tell the module to be ready to acquire data; reset the module's progress to 0.0.
    scopeModule.execute()

    # Enable the scope: Now the scope is ready to record data upon receiving triggers.
    daq.setInt("/%s/scopes/0/enable" % device, 1)
    daq.sync()

    start = time.time()
    timeout = 30  # [s]
    records = 0
    progress = 0
    # Wait until the Scope Module has received and processed the desired number of records.
    while (records < num_records) or (progress < 1.0):
        time.sleep(0.5)
        records = scopeModule.getInt("records")
        progress = scopeModule.progress()[0]
        print(
            f"Scope module has acquired {records} records (requested {num_records}). "
            f"Progress of current segment {100.0 * progress}%.",
            end="\r",
        )
        # Advanced use: It's possible to read-out data before all records have been recorded
        # (or even before all segments in a multi-segment record have been recorded). Note that
        # complete records are removed from the Scope Module and can not be read out again; the
        # read-out data must be managed by the client code. If a multi-segment record is read-out
        # before all segments have been recorded, the wave data has the same size as the complete
        # data and scope data points currently unacquired segments are equal to 0.
        #
        # data = scopeModule.read(True)
        # wave_nodepath = f"/{device}/scopes/0/wave"
        # if wave_nodepath in data:
        #   Do something with the data...
        if (time.time() - start) > timeout:
            # Break out of the loop if for some reason we're no longer receiving scope data from
            # the device.
            print(
                f"\nScope Module did not return {num_records} records after {timeout} s - \
                    forcing stop."
            )
            break
        
    print("")
    daq.setInt("/%s/scopes/0/enable" % device, 0)

    # Read out the scope data from the module.
    data = scopeModule.read(True)

    # Stop the module; to use it again we need to call execute().
    scopeModule.finish()

    return data


def check_scope_record_flags(scope_records):
    """
    Loop over all records and print a warning to the console if an error bit in
    flags has been set.

    Warning: This function is intended as a helper function for the API's
    examples and it's signature or implementation may change in future releases.
    """
    num_records = len(scope_records)
    for index, record in enumerate(scope_records):
        if record[0]["flags"] & 1:
            print(
                f"Warning: Scope record {index}/{num_records} flag indicates dataloss."
            )
        if record[0]["flags"] & 2:
            print(
                f"Warning: Scope record {index}/{num_records} indicates missed trigger."
            )
        if record[0]["flags"] & 4:
            print(
                f"Warning: Scope record {index}/{num_records} indicates transfer failure \
                    (corrupt data)."
            )
        totalsamples = record[0]["totalsamples"]
        
        for wave in record[0]["wave"]:
            # Check that the wave in each scope channel contains the expected number of samples.
            assert (
                len(wave) == totalsamples
            ), f"Scope record {index}/{num_records} size does not match totalsamples."


input1 = 1 # "currin0", "current_input0": Current Input 1
input2 = 8 # "auxin0", "auxiliary_input0": Aux Input 1
stream_rate = 6 # "938_kHz": 938 kHz

#device_id = "dev3258"
device_id = "dev5236"

#server_host = "192.168.50.234"
server_host = "192.168.81.210"

server_port = 8004

apilevel = 6

(daq, device, _) = zhinst.utils.create_api_session(device_id, apilevel, server_host=server_host, server_port=server_port)

zhinst.utils.api_server_version_check(daq)

daq.setDebugLevel(3)

# %% Scope setup
zhinst.utils.disable_everything(daq, device)

clockbase = daq.getInt(f"/{device}/clockbase")
rate = clockbase / 2**stream_rate

daq.sync()

daq.setInt("/%s/scopes/0/length" % device, 16384)  # Length
daq.setInt("/%s/scopes/0/channel" % device, 3)  # 2Ch

daq.setInt("/%s/scopes/0/channels/*/bwlimit" % device, 1)

daq.setInt("/%s/scopes/0/channels/0/inputselect" % device, input1)
daq.setInt("/%s/scopes/0/channels/1/inputselect" % device, input2)

daq.setInt("/%s/scopes/0/time" % device, stream_rate)  # 938kHz

daq.setInt("/%s/scopes/0/single" % device, 0)  # continuos

daq.setInt("/%s/scopes/0/segments/enable" % device, 0)  # no segments

daq.setInt("/%s/currins/0/autorange" % device, 1)
# daq.setDouble('/dev3258/currins/0/range', 0.00000001)

daq.sync()

time.sleep(7) # Wait 2s for autorange to complete

scopeModule = daq.scopeModule()
scopeModule.set("mode", 1) # TODO: Check this!
scopeModule.set("averager/weight", 1)
scopeModule.set("historylength", 100)

daq.setInt("/%s/scopes/0/trigchannel" % device, input2) # input2 as trigger
daq.setInt("/%s/scopes/0/trigslope" % device, 1) # Rising Edge
daq.setDouble("/%s/scopes/0/triglevel" % device, 0.10000000) # 100mV
daq.setDouble("/%s/scopes/0/trighysteresis/mode" % device, 0)
daq.setDouble("/%s/scopes/0/trighysteresis/absolute" % device, 0.02000000) # 20mV hyst
daq.setInt("/%s/scopes/0/trigenable" % device, 1) # enable trigger

daq.setInt("/%s/scopes/0/triggate/enable" % device, 0)
daq.setDouble("/%s/scopes/0/trigreference" % device, 0.50000000)

daq.sync()


# %% Programm
wave_nodepath = f"/{device}/scopes/0/wave"
scopeModule.subscribe(wave_nodepath)

i = 0
while i < 2:
    i += 1
    
    data_with_trig = get_scope_records(device, daq, scopeModule, 100)

    assert (
        wave_nodepath in data_with_trig
        ), f"The Scope Module did not return data for {wave_nodepath}."

    print(
        f"Number of scope records returned with triggering enabled: \
        {len(data_with_trig[wave_nodepath])}."
        )
    
    check_scope_record_flags(data_with_trig[wave_nodepath])


    if True:
    
        # Get the instrument's ADC sampling rate.
        clockbase = daq.getInt(f"/{device}/clockbase")
    
        def plot_scope_records(axis, scope_records, scope_input_channel, scope_time=0):
            """
            Helper function to plot scope records.
            """
            colors = [
                cm.Blues(np.linspace(0, 1, len(scope_records))),
                cm.Greens(np.linspace(0, 1, len(scope_records))),
            ]
            for index, record in enumerate(scope_records):
                totalsamples = record[0]["totalsamples"]
                wave = record[0]["wave"][scope_input_channel, :]
    
                if not record[0]["channelmath"][scope_input_channel] & 2:
                    # We're in time mode: Create a time array relative to the trigger time.
                    dt = record[0]["dt"]
                    # The timestamp is the timestamp of the last sample in the scope segment.
                    timestamp = record[0]["timestamp"]
                    triggertimestamp = record[0]["triggertimestamp"]
                    t = np.arange(-totalsamples, 0) * dt + (timestamp - triggertimestamp) / float(clockbase)
                    axis.plot(1e6 * t, wave, color=colors[scope_input_channel][index])
    
            axis.grid(True)
            axis.set_ylabel("Amplitude [V]")
            axis.autoscale(enable=True, axis="x", tight=True)
    
        fig1, axis1 = plt.subplots()
    
        # Plot the scope data with triggering enabled.
        plot_scope_records(axis1, data_with_trig[wave_nodepath], 0)
        #plot_scope_records(axis1, data_with_trig[wave_nodepath], 1)
        axis1.axvline(0.0, linewidth=2, linestyle="--", color="k", label="Trigger time")
        axis1.set_title(f"{len(data_with_trig[wave_nodepath])} Scope records from 2 channels ({device}, triggering enabled)")
        axis1.set_xlabel("t (relative to trigger) [us]")
        fig1.show()
        
        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.save(starttime, data_with_trig)

print("end")


sys.exit()

# %% functions

pressed_key = -1

def on_press(event):
    global pressed_key
    pressed_key = event.key


def evaluate_scope_file_old(data, threasholds):
    global pressed_key
        
    Vdiv = 40
    
    filtered_peaks = []
    discard_peaks = []
    good_peaks = []
    
    parameters = []
    
    avg_len = -1
    

    t = data[0]
    dt = t[1]-t[0]
    ch1 = data[1]
    ch2 = data[2]
    
    st = hf.SchmittTrigger(0.01, 0.005)
    trigger = [st.process_input(abs(x)) for x in ch2]
    
    trigger = hf.preplace_ones(trigger, hf.time_to_samples(2E-4, "s", 1/dt))
    trigger = hf.reduce_ones(trigger, hf.time_to_samples(5E-4, "s", 1/dt))
    
    cond = abs(ch2) > np.max(abs(ch2))*0.96
    
    appV = np.average(abs(ch2[cond]))
   
    sliced_ch1 = hf.slice_array(ch1, trigger)
    sliced_t = hf.slice_array(t, trigger)
    
    if avg_len < 0:
        for bt in sliced_t:
            avg_len += len(bt)
            
        avg_len /= len(sliced_t)
    
    
    for sch1, st in zip(sliced_ch1, sliced_t):
        if len(st) > avg_len*0.7:
            
            deriv = hf.calculate_derivative(st, sch1)**2
            baseline = hf.interpolate_baseline(st, deriv)
            maxdb = np.max(deriv-baseline)
            
            samples_to_average = 16
            
            convol = np.convolve(sch1, np.ones(samples_to_average)/samples_to_average, mode='valid')
            diff = scipy.integrate.simpson(abs(sch1[:-(samples_to_average-1)]-convol),x=st[:-(samples_to_average-1)])
            
            threashold_spike, threashold_continus = threasholds[hf.closest_key(threasholds, appV*Vdiv)]
            
            if (maxdb > threashold_spike) or (diff > threashold_continus): # 1.8 to include sigular events 
               
                plt.plot(st, sch1, label=str(maxdb) + " " + str(diff*1E10))
                plt.draw()
                
                while not plt.waitforbuttonpress():
                    print("looping")
                    
                plt.cla()
                
                if pressed_key == " ":
                    print("KEEP:" + str(maxdb) + " " + str(diff*1E10) + " @ " + str(appV*Vdiv))
                    filtered_peaks.append([st, sch1])
                    good_peaks.append([st-st[0], abs(np.asarray(sch1))])
                    
                    parameters.append({"Vapp": appV*Vdiv, "Vraw": appV, "threashold_spike": threashold_spike, "threashold_continus": threashold_continus, "samples_to_average": samples_to_average})
                else:
                    discard_peaks.append([st-st[0], abs(np.asarray(sch1))])
                    print("DISCARD:" + str(maxdb) + " " + str(diff*1E10) + " @ " + str(appV*Vdiv))
            else:
                discard_peaks.append([st-st[0],abs(np.asarray(sch1))])
                
            pressed_key = -1
            time.sleep(0.1)
           
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if len(filtered_peaks) > 0:
              np.save("good" + starttime, hf.pad_arrays_to_4096(good_peaks))
          
    if len(discard_peaks) > 0:
              np.save("bad" +  starttime, hf.pad_arrays_to_4096(discard_peaks))
    
    return filtered_peaks, parameters


def evaluate_scope_file(data, threasholds):
    global pressed_key
    
    clockbase = 60E6
    Vdiv = 40
    
    filtered_peaks = []
    
    discard_peaks = []
    good_peaks = []
    
    parameters = []
    
    avg_len = -1
    
    for record in data.item()[wave_nodepath]:
    
        ch1 = record[0]["wave"][0, :]
        ch2 = record[0]["wave"][1, :]
        
        st = hf.SchmittTrigger(0.01, 0.005)
        trigger = [st.process_input(abs(x)) for x in ch2]
        
        
        totalsamples = record[0]["totalsamples"]
        dt = record[0]["dt"]
        
        trigger = hf.preplace_ones(trigger, hf.time_to_samples(2E-4, "s", 1/dt))
        trigger = hf.reduce_ones(trigger, hf.time_to_samples(5E-4, "s", 1/dt))
        
        timestamp = record[0]["timestamp"]
        triggertimestamp = record[0]["triggertimestamp"]
        
        t = np.arange(-totalsamples, 0) * dt + (timestamp - triggertimestamp) / float(clockbase)
        
        cond = abs(ch2) > np.max(abs(ch2))*0.96
        
        appV = np.average(abs(ch2[cond]))
        #print("V_app: " + str(appV*Vdiv))
        
        sliced_ch1 = hf.slice_array(ch1, trigger)
        sliced_t = hf.slice_array(t, trigger)
        
        if avg_len < 0:
            for bt in sliced_t:
                avg_len += len(bt)
                
            avg_len /= len(sliced_t)
        
        
        for sch1, st in zip(sliced_ch1, sliced_t):
            if len(st) > avg_len*0.7:
                
                deriv = hf.calculate_derivative(st, sch1)**2
                baseline = hf.interpolate_baseline(st, deriv)
                maxdb = np.max(deriv-baseline)
                
                samples_to_average = 16
                
                convol = np.convolve(sch1, np.ones(samples_to_average)/samples_to_average, mode='valid')
                diff = scipy.integrate.simpson(abs(sch1[:-(samples_to_average-1)]-convol),x=st[:-(samples_to_average-1)])
                
                threashold_spike, threashold_continus = threasholds[hf.closest_key(threasholds, appV*Vdiv)]
                
                if (maxdb > threashold_spike) or (diff > threashold_continus): # 1.8 to include sigular events 
                   
                    plt.plot(st, sch1, label=str(maxdb) + " " + str(diff*1E10))
                    plt.draw()
                    
                    while not plt.waitforbuttonpress():
                        print("looping")
                        
                    plt.cla()
                    
                    if pressed_key == " ":
                        print("KEEP:" + str(maxdb) + " " + str(diff*1E10) + " @ " + str(appV*Vdiv))
                        filtered_peaks.append([st, sch1])
                        good_peaks.append([st-st[0],abs(np.asarray(sch1))])
                        parameters.append({"Vapp": appV*Vdiv, "Vraw": appV, "threashold_spike": threashold_spike, "threashold_continus": threashold_continus, "samples_to_average": samples_to_average})
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


def view_file(data, title=""):
    
    clockbase = 60E6
    Vdiv = 41
    
    wave_nodepath = "/dev3258/scopes/0/wave"
    
    for key in data.item().keys():
        if "/dev" in key:
            wave_nodepath = key
    
    fig, axes = plt.subplots(1,2)
    
    if title:
        fig.suptitle(title)
    
    for record in data.item()[wave_nodepath]:
        
        ch1 = record[0]["wave"][0, :]
        ch2 = record[0]["wave"][1, :] * Vdiv
        
        totalsamples = record[0]["totalsamples"]
        dt = record[0]["dt"]
        
        timestamp = record[0]["timestamp"]
        triggertimestamp = record[0]["triggertimestamp"]
        
        t = np.arange(-totalsamples, 0) * dt + (timestamp - triggertimestamp) / float(clockbase)
        
        axes[0].plot(t, ch1)
        axes[1].plot(t, ch2)
        

def view_file_proc(data):
    
    for dataset in data:
        plt.plot(dataset[0], dataset[1])
    

# %% test 0
folder = r"D:\Older PVDF Data\B3_9 bs 2.6Vpp"

ar1 = []
ar2 = []

plt.ion()
plt.connect("key_press_event", on_press)


for file in os.listdir(folder):
    filename, ext=os.path.splitext(file)  
    
    #Skip all files that arent txt.
    if not ext == ".txt":
        continue
            
    try:    
        t, ch1, ch2, _ = hf.read_data(folder + '\\' + file)
        
        print("Eval'ing: " + str(filename))
        thresholds = {20: [2, 2E-10],
                      29: [2, 6.1E-10],
                      33: [6, 10.29E-10]}
        
        
        a1, a2 = evaluate_scope_file_old([t, ch1, ch2], thresholds)
        
        
        if len(a1) > 0:
            ar1.append(a1)
            ar2.append(a2)
        
    except UnicodeDecodeError as ude:
        print(ude)
        continue


starttime = datetime.now().strftime("%Y-%m-%d_%H-%M")

flatar1 = [item for sublist in ar1 for item in sublist]
flatar2 = [item for sublist in ar2 for item in sublist]

npar1 = np.asarray(flatar1, dtype="object")

np.savez_compressed("scope_eval_" + str(starttime), npar1, flatar2)

print("exit")


# %% test 1

wave_nodepath = "/dev3258/scopes/0/wave" 
folder = r"D:\Session 240215 PVDF\1. 10V\shows events"


plt.ion()
plt.connect("key_press_event", on_press)

ar1 = []
ar2 = []

for file in os.listdir(folder):
    filename, ext = os.path.splitext(file)
    
    #Skip all files that arent .npy.
    if not ext == ".npy":
        continue
    
    
    # Lower Voltage 1, 2.1E-10
    # Higher Voltage 7, 6E-10
    
    print("Eval'ing: " + str(filename))
    
    thresholdsPVDF = {20: [2, 2E-10],
                  29: [2, 6.1E-10],
                  33: [6, 10.29E-10]}
    
    
    thresholdsBTAC10 = {16: [0.1, 0.30]}
    
    a1, a2 = evaluate_scope_file(np.load(folder + '\\' + file, allow_pickle = True), thresholdsPVDF)
    
    #a1, a2 = evaluate_scope_file(np.load(folder + '\\' + file, allow_pickle = True), thresholds)
    
    
    if len(a1) > 0:
        ar1.append(a1)
        ar2.append(a2)


starttime = datetime.now().strftime("%Y-%m-%d_%H-%M")

flatar1 = [item for sublist in ar1 for item in sublist]
flatar2 = [item for sublist in ar2 for item in sublist]

npar1 = np.asarray(flatar1, dtype="object")

np.savez_compressed("scope_eval_" + str(starttime), npar1, flatar2)

print("exit")
    
# %% test 2

data = np.load(r"\\netfilec.ad.uni-heidelberg.de\home\a\av179\Desktop\Experimental eval\data_scope\Session 240209 B3_9 sb\scope_eval_2024-02-13_15-12.npz", allow_pickle=True)

data_old1 = np.load(r"P:\Desktop\Experimental eval\data_scope\Older Data\B3_9 bs\scope_eval_2024-02-13_17-32.npz", allow_pickle=True)
data_old2 = np.load(r"P:\Desktop\Experimental eval\data_scope\Older Data\B3_9 bs 2.6Vpp\scope_eval_2024-02-13_17-36.npz", allow_pickle=True)



arr1 = data["arr_0"]

arr1 = np.append(arr1, data_old1["arr_0"], axis=0)
arr1 = np.append(arr1, data_old2["arr_0"], axis=0)

arr2 = data["arr_1"]

arr2 = np.append(arr2, data_old1["arr_1"], axis=0)
arr2 = np.append(arr2, data_old2["arr_1"], axis=0)


for x,y in arr1:
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    if (np.sum([y>0]) - np.sum([y<0])) > 0:
        plt.plot(x-x[0], y)
        
    else:
        plt.plot(x-x[0], y)
    
    
print("exit")

# %% testo

numbers = [x["Vapp"] for x in arr2]
groups = hf.group_numbers_within(numbers, 0.5, True)

#print([item[0] for item in groups])

fig, axes = plt.subplots(3,3)

for i, group in enumerate(groups):
    
    axes[i // 3, i % 3].set_title(f"{group[0]:.1f} V")
    
    for j in group[1]:
        axes[i // 3, i % 3].plot(arr1[j][0]-arr1[j][0][0],arr1[j][1])
        
    
# %%test^4
def plot_group(group, ax = None):
    
    bhdata = []
    
    for i in range(len(group[1])):
        
        data_t = arr1[group[1][i]][0] - arr1[group[1][i]][0][0]
        data_y = np.asarray(arr1[group[1][i]][1])
        absy = abs(data_y)
        
        #plt.plot(data_t, data_y)
        #plt.show()
        
        #plt.plot(data_t, absy)
        #plt.show()
        
        bhdata.extend(bhm.analyzeBH(data_t, absy, 1E-5))
        
    #bhm.analyze(bhdata, xmin=6e-3, xmax=1E2, binnumber=50)
    
    edges, hist = powerlaw.pdf(bhdata, number_of_bins = 50)
    
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    fit = powerlaw.Fit(bhdata)
    fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha, 2), round(fit.sigma, 2)), ax = ax)
    ax.scatter(bin_centers, hist)
    ax.legend()
    ax.set_title(f"{group[0]:.1f} V")
    
    return fit.alpha


fig2, axes2 = plt.subplots(3,3)

alphas = []

for i, group in enumerate(groups):
    alphas.append(plot_group(group, axes2[i // 3, i % 3]))
    
    
plt.show()
plt.plot([x[0] for x in groups], alphas)



# %% plot all files
folder = r"D:\Session 240215 PVDF\2. 15V\\"

for file in os.listdir(folder):
    filename, ext = os.path.splitext(file)
    
    #Skip all files that arent .npy.
    if not ext == ".npy":
        continue
    
    view_file(np.load(folder + file, allow_pickle=True), filename)

    
