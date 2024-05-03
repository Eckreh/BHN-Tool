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
import torch.nn as nn
import torch
import shutil
from mpl_axes_aligner import align


plt.rcParams.update({'font.size': 18})
# plt.rcParams.update({'figure.autolayout': True})

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

# %% Connection


device_id = "dev3258"
# device_id = "dev5236"

server_host = "192.168.50.234"
# server_host = "192.168.81.210"

server_port = 8004

apilevel = 6

(daq, device, _) = zhinst.utils.create_api_session(device_id,
                                                   apilevel, server_host=server_host, server_port=server_port)

zhinst.utils.api_server_version_check(daq)

daq.setDebugLevel(3)


# %% Scopesetup

input1 = 1  # "currin0", "current_input0": Current Input 1
input2 = 0  # "auxin0", "auxiliary_input0": Aux Input 1

stream_rate = 6  # 6 = "938_kHz": 938 kHz 60MHz Clock / 2^6 = 938kHz


zhinst.utils.disable_everything(daq, device)

clockbase = daq.getInt(f"/{device}/clockbase")
rate = clockbase / 2**stream_rate

daq.sync()

daq.setInt("/%s/scopes/0/length" % device, int(2**14))  # Length 2^14 = 16386
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

time.sleep(4)  # Wait 5s for autorange to complete

scopeModule = daq.scopeModule()
scopeModule.set("mode", 1)  # TODO: Check this!
scopeModule.set("averager/weight", 1)
scopeModule.set("historylength", 100)

# daq.setInt("/%s/scopes/0/trigchannel" % device, input2) # input2 as trigger
daq.setInt('/dev3258/scopes/0/trigchannel', 2)
daq.setInt("/%s/scopes/0/trigslope" % device, 1)  # Rising Edge
daq.setDouble("/%s/scopes/0/triglevel" % device, 0.10000000)  # 100mV
daq.setDouble("/%s/scopes/0/trighysteresis/mode" % device, 0)
daq.setDouble("/%s/scopes/0/trighysteresis/absolute" %
              device, 0.02000000)  # 20mV hyst
daq.setInt("/%s/scopes/0/trigenable" % device, 1)  # enable trigger

daq.setInt("/%s/scopes/0/triggate/enable" % device, 0)
daq.setDouble("/%s/scopes/0/trigreference" % device, 0.50000000)

daq.sync()


# %% Programm
wave_nodepath = f"/{device}/scopes/0/wave"
scopeModule.subscribe(wave_nodepath)

i = 0
while i < 100:
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
                    t = np.arange(-totalsamples, 0) * dt + \
                        (timestamp - triggertimestamp) / float(clockbase)
                    axis.plot(1e6 * t, wave,
                              color=colors[scope_input_channel][index])

            axis.grid(True)
            axis.set_ylabel("Amplitude [V]")
            axis.autoscale(enable=True, axis="x", tight=True)

        fig1, axis1 = plt.subplots()

        # Plot the scope data with triggering enabled.
        plot_scope_records(axis1, data_with_trig[wave_nodepath], 0)
        # plot_scope_records(axis1, data_with_trig[wave_nodepath], 1)
        axis1.axvline(0.0, linewidth=2, linestyle="--",
                      color="k", label="Trigger time")
        axis1.set_title(
            f"{len(data_with_trig[wave_nodepath])} Scope records from 2 channels ({device}, triggering enabled)")
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


def view_file(data, title="", fig=None, axes=None, ch2ax=None):

    clockbase = 60E6
    # ALSO HERE REMEMBER
    Vdiv = 41

    wave_nodepath = "/dev3258/scopes/0/wave"

    for key in data.item().keys():
        if "/dev" in key:
            wave_nodepath = key

    if not fig and not axes:
        fig, axes = plt.subplots(1, 1)

    if len(title) > 0:
        fig.suptitle(title)

    for record in data.item()[wave_nodepath]:

        ch1 = record[0]["wave"][0, :]
        ch2 = record[0]["wave"][1, :] * Vdiv

        totalsamples = record[0]["totalsamples"]
        dt = record[0]["dt"]

        timestamp = record[0]["timestamp"]
        triggertimestamp = record[0]["triggertimestamp"]

        t = np.arange(-totalsamples, 0) * dt + (timestamp -
                                                triggertimestamp) / float(clockbase)

        axes.plot(t, ch1)

        if ch2ax:
            ch2ax.plot(t, ch2)
            align.yaxes(axes, 0, ch2ax, 0, 0.5)


def view_file_proc(data):

    for dataset in data:
        plt.plot(dataset[0], dataset[1])


def coarse_sieving(folder, view_ch2=False, viewonly=False):
    global pressed_key

    fig, axes = plt.subplots(1, 1)

    if view_ch2:
        ch2ax = axes.twinx()
        ch2ax.set_prop_cycle('color', plt.cm.Greens(np.linspace(0, 1, 100)))
    else:
        ch2ax = None

    plt.connect("key_press_event", on_press)

    for file in os.listdir(folder):
        filename, ext = os.path.splitext(file)

        if not ext == ".npy":
            continue

        view_file(np.load(os.path.join(folder, file), allow_pickle=True), str(
            filename), fig, axes, ch2ax=ch2ax)
        plt.draw()

        while not plt.waitforbuttonpress():
            print("looping")

        if view_ch2:
            ch2ax.clear()
            ch2ax.set_prop_cycle(
                'color', plt.cm.Greens(np.linspace(0, 1, 100)))

        axes.clear()

        if viewonly:
            continue

        if not os.path.exists(os.path.join(folder, "fail")):
            os.makedirs(os.path.join(folder, "fail"))

        if not os.path.exists(os.path.join(folder, "pass")):
            os.makedirs(os.path.join(folder, "pass"))

        if pressed_key == " ":
            print(f"PASS: {file}")
            shutil.move(os.path.join(folder, file),
                        os.path.join(folder, 'pass', file))
        else:
            print(f"FAIL: {file}")
            shutil.move(os.path.join(folder, file),
                        os.path.join(folder, 'fail', file))

        pressed_key = -1


def power_law(x, a1, b1):
    return b1*x**(-a1)


def lin(x, m, b):
    return (-m)*x+b

def getxy(group):

    bhdata = []

    for i in range(len(group[1])):

        data_t = arr1[group[1][i]][0] - arr1[group[1][i]][0][0]
        data_y = np.asarray(arr1[group[1][i]][1])

        bhdata.extend(bhm.analyzeBH(data_t, data_y, 1E-5))

    edges, hist = powerlaw.pdf(bhdata, number_of_bins=100)
    bin_centers = (edges[1:] + edges[:-1]) / 2.0

    return bin_centers, hist


def calculate_peak_width(y, peak_index):
    peak_height = y[peak_index]
    half_max_height = np.mean(y)

    # Find left and right indexes where y crosses half-maximum
    try:
        left_index = np.where(y[:peak_index] < half_max_height)[0][-1]
    except IndexError:
        print("How does this happen?")
    
    try:
        right_index = np.where(y[peak_index:] < half_max_height)[0][0] + peak_index
    except IndexError:
        right_index = len(y) - 1

    return left_index, right_index

def plot_group(group, ax=None, addon="", dofit=True, xmin=None, xmax=None):

    bhdata = []
    bhdata_peak = []
    
    rts = []
    for i in range(len(group[1])):

        data_t = arr1[group[1][i]][0] - arr1[group[1][i]][0][0]
        data_y = np.asarray(arr1[group[1][i]][1])
        
        peaks, _ = scipy.signal.find_peaks(abs(data_y), width=30, prominence=0.7*max(abs(data_y)))
        
        
        if len(peaks) == 1:
            li, re = calculate_peak_width(abs(data_y), peaks[0])
            
            #plt.plot(data_t, data_y)
            #plt.hlines(0, data_t[li], data_t[re], color="Black")
            if abs(re-li) > 3:
                bhdata_peak.extend(bhm.analyzeBH(data_t[li:re], data_y[li:re], 1E-4))
        
        
        
        # absy = abs(data_y)
        rts.append(float(arr2[group[1][i]]["risetime"]))

        # plt.plot(data_t, data_y)
        # plt.show()

        # plt.plot(data_t, absy)
        # plt.show()

        bhdata.extend(bhm.analyzeBH(data_t, data_y, 1E-4))
        
        

    # bhm.analyze(bhdata, xmin=6e-3, xmax=1E2, binnumber=50)
    
    edges, hist = powerlaw.pdf(bhdata)
    edges_pk, hist_pk = powerlaw.pdf(bhdata_peak)

    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    bin_centers_pk = (edges_pk[1:] + edges_pk[:-1]) / 2.0
    
    if dofit:
        fit = powerlaw.Fit(bhdata, estimate_discrete=False, xmin=xmin, xmax=xmax, fit_method="Likelihood")
        
        fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$ {}".format(
            round(fit.alpha, 2), round(fit.sigma, 2), addon), ax=ax)
        
        xmin = fit.xmin
        
        if xmax is None:
            xmax = fit.xmax

    # p0 = [1.5, 1]
    # popt, pcov = scipy.optimize.curve_fit(power_law, bin_centers, hist, p0=p0)

    # print(popt)
    
    print(f"{(np.mean(rts)*1E6):.1f}, {(np.max(rts)*1E6):.1f} , {(np.min(rts)*1E6):.1f}", "us")

    if not ax is None:
        ax.scatter(bin_centers, hist)
        ax.scatter(bin_centers_pk, hist_pk)
        
        if xmax is None or xmin is None:    
            logx = np.log10(bin_centers, where=bin_centers > 0)
            logy = np.log10(hist, where=hist > 0)
        else:
            cond = np.logical_and(bin_centers > xmin, bin_centers < xmax)
            logx = np.log10(bin_centers[cond], where=bin_centers[cond] > 0)
            logy = np.log10(hist[cond], where=hist[cond] > 0)
        
        
        popt,pcov = scipy.optimize.curve_fit(lin, logx, logy, p0=[-1.5, 1])
        
        if xmax is None or xmin is None:
            ax.plot(bin_centers, power_law(bin_centers, popt[0], 10**popt[1]),label=f"α_Lin = {popt[0]:.2f}")
        else:
            ax.plot(bin_centers[cond], power_law(bin_centers[cond], popt[0], 10**popt[1]),label=f"α_Lin = {popt[0]:.2f}")
            
        
        # ax.plot(bin_centers, power_law(bin_centers, *popt), label=r"$\alpha_{lin} = $" + f"{popt[0]:.2f}")
        ax.legend()
        
        if isinstance(group[0], str):
            ax.set_title(group[0])
        else:
            ax.set_title(f"{group[0]:.1f} V")

        plt.ylabel(r'probability P($S=S_i$)')
        plt.xlabel(r'Slew-Rate $S=\left[\frac{A^2}{s^2}\right]$')

    if dofit:
        return fit.alpha, popt[0]

    return None


def load_files(files):

    arr0 = []
    arr1 = []

    for file in files:
        data = np.load(file, allow_pickle=True)

        if len(arr1) == 0:
            arr0 = data["arr_0"]
            arr1 = data["arr_1"]
        else:
            arr0 = np.append(arr0, data["arr_0"], axis=0)
            arr1 = np.append(arr1, data["arr_1"], axis=0)

    return arr0, arr1

# for x, y in arr1:
#     x = np.asarray(x)
#     y = np.asarray(y)

#     if (np.sum([y > 0]) - np.sum([y < 0])) > 0:
#         plt.plot(x-x[0], y)

#     else:
#         plt.plot(x-x[0], y)

# %% testo 1


files = [r"D:\Older PVDF Data\B3_9 bs 2.6Vpp\scope_eval_2024-04-03_15-32.npz",
         r"D:\Older PVDF Data\B3_9 bs\scope_eval_2024-04-03_15-23.npz",
         r"D:\Session 240209 B3_9 sb\scope_eval_2024-04-04_15-59.npz",
         r"D:\Session 240215 PVDF\1. 10V\scope_eval_2024-04-05_11-29.npz",
         r"D:\Session 240215 PVDF\2. 15V\scope_eval_2024-04-05_13-25.npz",
         r"D:\Session 240215 PVDF\3. 20V\scope_eval_2024-04-05_13-56.npz",
         r"D:\Session 240215 PVDF\3. 25V\scope_eval_2024-04-08_16-29.npz",
         r"D:\Session 240215 PVDF\6. 30V\scope_eval_2024-04-08_17-04.npz",
         r"D:\Session 240215 PVDF\7. 30V\scope_eval_2024-04-08_17-13.npz",
         r"D:\Session 240215 PVDF\8. 30V\scope_eval_2024-04-09_13-39.npz"]



arr1, arr2 = load_files(files)


xmin_arr = [2E-3, 3E-3, 1E0, 3E-3, 1E0, 2E0, 1E-1, 5E-3, 4E-3, 1E-2]
xmax_arr = [1E0,  1E2,  5E2, 1E1,  5E2, 3E2, 3E1,  1E0,  1E0,  1E1]

skip = 0

numbers = [x["Vapp"] for x in arr2]
groups = hf.group_numbers(numbers, 0.5, True)

results1 = []

plt.plot(arr1[-1][0],arr1[-1][1])


for xmin,xmax,group in zip(xmin_arr,xmax_arr,groups):
    
    if skip > 0:
        skip -= 1
        continue
    
    fig,ax = plt.subplots(1,1)
    results1.append(plot_group(group, ax, xmin=xmin, xmax=xmax))


print("exit")


# %% newnewnewn


arr1, arr2 = load_files([r"D:\Session 240220\B3_5 _bs\70\scope_eval_2024-04-09_14-03.npz",
                         r"D:\Session 240220\B3_5 _bs\120\scope_eval_2024-04-10_15-25.npz"])

numbers = [x["Vapp"] for x in arr2]
groups = hf.group_numbers(numbers, 0.5, True)

results2 = []

xmin_arr = [1E-3, 1E-3, 1E-2, 1E-2, 1E-2, 1E0, 1E0, 3E-3, 1E-1]
xmax_arr = [1E-1, 1E0, 1E2, 1E2, 1E2, 5E2, 2E2, 3E-1, 1E1]

skip = 0

for xmin,xmax,group in zip(xmin_arr,xmax_arr,groups):
    
    if skip > 0:
        skip -= 1
        continue
    
    fig,ax = plt.subplots(1,1)
    results2.append(plot_group(group, ax, xmin=xmin, xmax=xmax))


# %% newnewnewnewnewnew

arr1, arr2 = load_files([r"D:\Session 240220\B3_5 _bs\70\scope_eval_2024-04-09_14-03.npz",
                         r"D:\Session 240220\B3_5 _bs\120\scope_eval_2024-04-10_15-25.npz"])

numbers = [x["Vapp"] for x in arr2]
groups = hf.group_numbers(numbers, 0.5, True)

results3 = []

xmin_arr = [1E-3, 1E-3,1E-2]
xmax_arr = [1E-1, 1E0, 1E2]


for xmin,xmax,group in zip(xmin_arr,xmax_arr,groups):
    
    if skip > 0:
        skip -= 1
        continue
    
    fig,ax = plt.subplots(1,1)
    results3.append(plot_group(group, ax, xmin=xmin, xmax=xmax))



# %% what da noise

arr1, arr2 = load_files([r"D:\Session 240319\Keysight 33600A  ANDERER DIVISIOR!! 10M stat 1M\amplifier\scope_eval_2024-04-24_13-45.npz"])

plt.plot(arr1[2][0],arr1[2][1])
numbers = [x["Vapp"] for x in arr2]
groups = hf.group_numbers(numbers, 0.5, True)


fig,ax = plt.subplots(1,1)
plot_group(groups[0], ax, xmin=3E-5, xmax=2E-4)

# %% group and plot



newgroups = []

for group in groups:

    length = len(group[1])

    max_part_length = 15
    min_part_length = 10

    num_parts = (len(group[1]) + max_part_length - 1) // max_part_length
    target_length = length // num_parts

    if target_length < min_part_length:
        target_length = min_part_length

    start = 0
    for i in range(num_parts):
        end = min(start + target_length, length)
        # split_parts.append(group[1][start:end])
        newgroups.append([group[0], group[1][start:end]])
        start = end


for g in groups:
    print(g[0], " -> ", len(g[1]))


fig, axes = plt.subplots(4, 3)

alphas = []

#for i, group in enumerate(newgroups):
#    # axes[i // 3, i % 3]
#    alphas.append(plot_group(group))


#toni100usX = [22, 25, 29]
#toni100usY = [1.53, 1.52, 1.52]
#toni200usY = [1.60, 1.49, 1.49]


#x = [g[0] for g in newgroups]
#y = alphas


#plt.scatter(x, y)
#plt.hlines(1.5, min(x), max(x), color="green")
#plt.vlines(25, min(y), max(y), color="red")

# xnonlow = [group[0] if len(group[1]) > 20 else 0 for group in groups]
# ynonlow = [alphas[i] if len(groups[i][1]) > 20 else 0 for i in range(len(alphas))]

# peaktypegroups = []

# for gr in groups:
#    peaktypegroups.append([gr[0], hf.group_data_by_metadata(arr2[gr[1]], "peaktype", lambda x: x[:2])])


# peakgroups = [hf.group_data_by_metadata(arr2[gr[1]], "peaktype") for gr in groups]


# for ptg in peaktypegroups:
#    fig, ax = plt.subplots(1,1)
#    plot_group([ptg[0], ptg[1]['LP']], ax, f"LP {len(ptg[1]['LP'])}")
#    plot_group([ptg[0], ptg[1]['SP']], ax, f"SP {len(ptg[1]['SP'])}")


# for mean, indices in groups:
#     print(mean, str(len(indices)))

# if len(groups) == 1:
#     fig, axes = plt.subplots(1,1)
#     plot_group(groups[0], axes)

# else:
#     for group in groups:
#         fig, axes = plt.subplots(1,1)
#         plot_group(group, axes)


# %%test^4

# MOVE PLOT GROUP FUNC To funcs


# fig2, axes2 = plt.subplots(3,3)

# alphas = []

# for i, group in enumerate(groups):
#    alphas.append(plot_group(group, axes2[i // 3, i % 3]))


# plt.show()
# plt.plot([x[0] for x in groups], alphas)


# %% plot all files
folder = r"D:\Session 240215 PVDF\2. 15V\\"

for file in os.listdir(folder):
    filename, ext = os.path.splitext(file)

    # Skip all files that arent .npy.
    if not ext == ".npy":
        continue

    view_file(np.load(folder + file, allow_pickle=True), filename)


# =============================================================================
# scope.subscribe('/dev3258/scopes/0/wave')
# scope.execute()
# # To read the acquired data from the module, use a
# # while loop like the one below. This will allow the
# # data to be plotted while the measurement is ongoing.
# # Note that any device nodes that enable the streaming
# # of data to be acquired, must be set before the while loop.
# # result = 0
# # while scope.progress() < 1.0 and not scope.finished():
# #     time.sleep(1)
# #     result = scope.read()
# #     print(f"Progress {float(scope.progress()) * 100:.2f} %\r")
# scope.finish()
# scope.unsubscribe('*')
#
# =============================================================================

# %% Anton

fig, axes = plt.subplots(1, 1)
fig2, axes2 = plt.subplots(1, 1)


ax2 = axes.twinx()

ax2.set_prop_cycle('color', plt.cm.Greens(np.linspace(0.7, 1, 5)))

wave_nodepath = "/dev3258/scopes/0/wave"

#data = np.load(
#    r"D:\Session 240319\PVDF B3_9 sb now shorted\long meaasurement\pass\2024-03-19_15-34-06.npy", allow_pickle=True)


## DATA For DWM exmaple with offset 0.014285
#data = np.load(r"D:\Session 240220\B3_bm\DWM70\2024-02-20_13-44-23.npy", allow_pickle=True)
data = np.load(r"D:\Session 240319\PVDF B3_9 sb now shorted\long meaasurement\fail\2024-03-19_15-35-18.npy", allow_pickle=True)

num = -1

for record in data.item()[wave_nodepath]:

    num += 1
    # and (num != 50 or num != 63 or num != 65 or num != 68)

    if (num < 90 or num > 90):  # and num != 2
        continue

    ch1 = record[0]["wave"][0, :]

    totalsamples = record[0]["totalsamples"]
    dt = record[0]["dt"]

    timestamp = record[0]["timestamp"]
    triggertimestamp = record[0]["triggertimestamp"]

    t = np.arange(-totalsamples, 0) * dt + \
        (timestamp - triggertimestamp) / float(60E6)

    # axes.plot(t, ch1, label=str(num))
   

    if len(record[0]["wave"]) == 2:
        ch2 = record[0]["wave"][1, :] * 41
        axes.plot(t, ch2, label=f"voltage {num}",color="tab:blue", zorder=1)
        axes.plot(t+0.016667, ch2, label=f"voltage {num}",color="tab:blue", zorder=1)
        axes.set_ylabel("Voltage [V]")
        
        

    ax2.plot(t, ch1, label=f"current {num}",color="tab:red", zorder=2)
    ax2.plot(t+0.016667, ch1, label=f"current {num}",color="tab:red", zorder=2)
    
    
    # jerks = hf.calculate_derivative(t, ch1)**2
    # bl = hf.interpolate_baseline(t, jerks)

    # pl2 = ax2.plot(t, jerks-bl, color="green", label="slew-rate")

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Current [A]")

    #ax2.set_ylabel("slew rate [A^2/s^2]")
    # f, psd = scipy.signal.periodogram(x, fs)
    # frequencies, psd = axes2.psd(ch2, Fs=1/dt, label=f"measurement {num}")

    # bindata = bhm.analyzeBH(t, ch1, 1E-5)
    # edges, hist = powerlaw.pdf(bindata, number_of_bins = 50)
    # bin_centers = (edges[1:] + edges[:-1]) / 2.0

    # axes.scatter(bin_centers, hist, label="Measurement")
   # axes2.set_ylabel(r'probability P($S=S_i$)')
    # axes2.set_xlabel(r'Slew-Rate $S=\left[\frac{A^2}{s^2}\right]$')

    # fit = powerlaw.Fit(bindata)
    # fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha, 2), round(fit.sigma, 2)), ax = axes2)

    # axes2.set_xlabel('Frequency (Hz)')
    # axes2.set_ylabel('Power/Frequency (dB/Hz)')
    # axes2.grid(True)

    align.yaxes(axes, 0, ax2, 0, 0.5)
    # pls = pl1+pl2
    # labs = [pl.get_label() for pl in pls]
    # axes.legend(pls, labs)

# axes.legend()


axes.set_ylabel("E-Field [V/m]", color="tab:blue")
ax2.set_ylabel("Current Density [A/m^2]", color="tab:red")
ax2.spines["right"].set_color("tab:red")
ax2.spines["left"].set_color("tab:blue")
ax2.set_yticks([])
axes.set_yticks([])
axes.set_xticks([])


axes2.legend()


# %% compare things

fig, axes = plt.subplots(2, 1)
ax2 = axes[0].twinx()
fig2, axes2 = plt.subplots(1, 1)
fig3, axes3 = plt.subplots(1, 1)

fig4, axes4 = plt.subplots(2, 1)

wave_nodepath = "/dev3258/scopes/0/wave"


data1 = np.load(r"D:\Session 240312 AFG\keysight33600A\Diff\2024-03-12_14-59-11.npy", allow_pickle=True)  # green
data2 = np.load(r"D:\Session 240312 AFG\3052C\Diff\2024-03-12_15-13-28.npy", allow_pickle=True)  # red
data3 = np.load(r"D:\Session 240312 AFG\1062\2024-03-12_15-27-59.npy", allow_pickle=True)  # blue
num = -1

rms1 = []
rms2 = []
rms3 = []

for record1, record2, record3 in zip(data1.item()[wave_nodepath], data2.item()[wave_nodepath], data3.item()[wave_nodepath]):

    num += 1

    if (num < 0 or num > 100):
        continue

    ch11 = record1[0]["wave"][0, :]

    totalsamples1 = record1[0]["totalsamples"]
    dt1 = record1[0]["dt"]
    timestamp1 = record1[0]["timestamp"]
    triggertimestamp1 = record1[0]["triggertimestamp"]
    t1 = np.arange(-totalsamples1, 0) * dt1 + \
        (timestamp1 - triggertimestamp1) / float(60E6)

    ch12 = record2[0]["wave"][0, :]
    totalsamples2 = record2[0]["totalsamples"]
    dt2 = record2[0]["dt"]
    timestamp2 = record2[0]["timestamp"]
    triggertimestamp2 = record2[0]["triggertimestamp"]
    t2 = np.arange(-totalsamples2, 0) * dt2 + \
        (timestamp2 - triggertimestamp2) / float(60E6)

    ch13 = record3[0]["wave"][0, :]
    totalsamples3 = record3[0]["totalsamples"]
    dt3 = record3[0]["dt"]
    timestamp3 = record3[0]["timestamp"]
    triggertimestamp3 = record3[0]["triggertimestamp"]
    t3 = np.arange(-totalsamples3, 0) * dt3 + \
        (timestamp3 - triggertimestamp3) / float(60E6)

    # axes.plot(t, ch1, label=str(num))
    axes[0].plot(t1, ch11, label=f"current {num}_1", color="green")
    axes[0].plot(t2, ch12, label=f"current {num}_2", color="red")
    axes[0].plot(t3, ch13, label=f"current {num}_2", color="blue")

    if len(record1[0]["wave"]) == 2 and len(record2[0]["wave"]) == 2:

        ch21 = record1[0]["wave"][1, :] * 41
        ch22 = record2[0]["wave"][1, :] * 41
        ch23 = record3[0]["wave"][1, :] * 41

        # ax2.plot(t1, ch21, label = f"voltage {num}_1", color = "green")
        # ax2.plot(t2, ch22, label = f"voltage {num}_2", color = "red")
        # ax2.plot(t2, ch23, label = f"voltage {num}_2", color = "blue")
        # ax2.set_ylabel("Voltage [V]")

        st = hf.SchmittTrigger(8, 7)

        trigger1 = [st.process_input(abs(x)) for x in ch21]
        st.reset()
        trigger2 = [st.process_input(abs(x)) for x in ch22]
        st.reset()
        trigger3 = [st.process_input(abs(x)) for x in ch23]

        trigger1 = hf.reduce_ones(
            trigger1, hf.time_to_samples(0.00012, "s", 1/dt1))
        trigger2 = hf.reduce_ones(
            trigger2, hf.time_to_samples(0.00012, "s", 1/dt2))
        trigger3 = hf.reduce_ones(
            trigger3, hf.time_to_samples(0.00012, "s", 1/dt2))

        sliced_ch11 = hf.slice_array(ch11, trigger1)
        sliced_t1 = hf.slice_array(t1, trigger1)

        sliced_ch12 = hf.slice_array(ch12, trigger2)
        sliced_t2 = hf.slice_array(t2, trigger2)

        sliced_ch13 = hf.slice_array(ch13, trigger3)
        sliced_t3 = hf.slice_array(t3, trigger3)

        skipsamples = hf.time_to_samples(3E-4, "s", 1/dt1)

        axes4[0].plot(sliced_t1[1][skipsamples:], sliced_ch11[1]
                      [skipsamples:], color="red", label="normal setup")
        
        axes4[0].plot(sliced_t2[1][skipsamples:], sliced_ch12[1]
                      [skipsamples:], color="green", label="no input")
        
        axes4[0].plot(sliced_t3[1][skipsamples:], sliced_ch13[1][skipsamples:],
                      color="blue", label="without amplifier")
        
        
        
        #mean1 = np.mean(sliced_ch11[1][skipsamples:])
        #mean2 = np.mean(sliced_ch12[1][skipsamples:])
        #mean3 = np.mean(sliced_ch13[1][skipsamples:])
        
        #rmstemp1 = np.sqrt(np.mean(np.square(sliced_ch11[1][skipsamples:] - mean1)))
        #rmstemp2 = np.sqrt(np.mean(np.square(sliced_ch12[1][skipsamples:] - mean2)))
        #rmstemp3 = np.sqrt(np.mean(np.square(sliced_ch13[1][skipsamples:] - mean3)))
        
        rms1.append(np.std(sliced_ch11[1][skipsamples:]))
        rms2.append(np.std(sliced_ch12[1][skipsamples:]))
        rms3.append(np.std(sliced_ch13[1][skipsamples:]))
        
        axes4[0].set_ylabel("Current [A]")
        axes4[0].set_xlabel("Time [s]")
        axes4[0].legend()

        axes4[1].psd(sliced_ch11[1][skipsamples:], Fs=1/dt1, color="green")
        axes4[1].psd(sliced_ch12[1][skipsamples:], Fs=1/dt2, color="red")
        axes4[1].psd(sliced_ch13[1], Fs=1/dt3, color="blue")

    fftch11 = np.fft.rfft(ch11)
    fftch12 = np.fft.rfft(ch12)

    freq1 = np.fft.rfftfreq(len(ch11), dt1)
    freq2 = np.fft.rfftfreq(len(ch12), dt2)

    # axes2.plot(freq1, np.abs(fftch11)**2, color="green")
    # axes2.plot(freq2, np.abs(fftch12)**2, color="red")

    axes[1].psd(ch11, Fs=1/dt1, color="green")
    axes[1].psd(ch12, Fs=1/dt2, color="red")
    axes[1].psd(ch13, Fs=1/dt3, color="blue")

    # axes2.magnitude_spectrum(sliced_ch13[1],Fs=1/dt3, color="blue")

    # axes2.set_yscale('log')
    # axes2.set_xscale('log')
    # axes2.set_xlim(0, 6E3)

    # axes3.plot(sliced_t1[1], sliced_ch11[1],color="green")
    # axes3.plot(sliced_t2[1], sliced_ch12[1],color="red")
    # axes3.plot(sliced_t3[1], sliced_ch13[1],color="blue")

    # frequencies, psd = axes3.psd(ch11, Fs=1/dt1, label=f"measurement {num}_1", color="green")
    # frequencies, psd = axes3.psd(ch12, Fs=1/dt2, label=f"measurement {num}_2", color="red")

    axes2.set_xlabel('Frequency (Hz)')
    axes2.set_ylabel('Power/Frequency (dB/Hz)')
    axes2.grid(True)

    # axes3.set_xlabel('Frequency (Hz)')
    # axes3.set_ylabel('Power/Frequency (dB/Hz)')
    # axes3.grid(True)

    # axes2.legend()
    # axes3.legend()

print("RMS1", np.mean(rms1))
print("RMS2", np.mean(rms2))
print("RMS3", np.mean(rms3))

# %% expectation

fig, axes = plt.subplots(2, 1)

frequency = 70

sr = 60E6
dur = 0.032


t = np.linspace(0, dur, int(dur*sr))
square_wave0 = 1/2 * np.sign(np.sin(2 * np.pi * frequency * 4 * t)) + 1/2
square_wave1 = 1/2 * np.sign(np.sin(2 * np.pi * frequency * t)) + 1/2
square_wave2 = 1/2 * np.sign(-np.sin(2 * np.pi * frequency * t)) + 1/2
pund = square_wave0*square_wave1 - square_wave0*square_wave2

pund *= 0.522


pundavg = np.convolve(pund, np.ones(105600)/105600, mode='same')

sos = scipy.signal.butter(2, 0.8, btype='low', output='sos')
filtered_signal = scipy.signal.sosfiltfilt(sos, pundavg)


axes[0].plot(t, pund)

# plt.plot(t, square_wave0*square_wave1)
# plt.plot(t, -square_wave0*square_wave2)

axes[0].plot(t, pundavg)
# plt.plot(t, filtered_signal)

# plt.plot(t, np.sin(2*np.pi*frequency*t))
# plt.plot(t, np.sin(2*np.pi*4*frequency*t))


# fft =  np.fft.rfft(pund)
# freq = np.fft.rfftfreq(len(pund), (t[1]-t[0]))
# plt.plot(freq, np.abs(fft)**2)

# plt.yscale('log')
# plt.xscale('log')
# plt.xlim((10, 10E6))

axes[1].psd(pund, Fs=sr)
axes[1].psd(pundavg, Fs=sr)

# axes[1].magnitude_spectrum(pund, Fs=sr)
# axes[1].magnitude_spectrum(pundavg, Fs=sr)

# %% random bs


bin_center_noise = [1.50003179e-05, 2.50005299e-05, 3.50007418e-05, 4.50009538e-05,
                    5.50011657e-05, 7.00014836e-05, 9.00019075e-05, 1.15002437e-04,
                    1.50003179e-04, 1.90004027e-04, 2.40005087e-04, 3.10006570e-04,
                    3.95008372e-04, 5.00010597e-04, 6.35013459e-04, 8.10017168e-04,
                    1.03002183e-03, 1.30502766e-03, 1.65503508e-03, 2.10004451e-03,
                    2.66505648e-03, 3.38007164e-03, 4.29009093e-03, 5.44511541e-03]


hist_noise = [7.29330153e+04, 1.72128394e+04, 4.91691428e+03, 1.66323451e+03,
              6.76940085e+02, 2.76599175e+02, 1.19192408e+02, 6.12642909e+01,
              2.09269112e+01, 8.18879136e+00, 2.72959712e+00, 2.04719784e+00,
              6.06577137e-01, 3.03288569e-01, 1.21315427e-01, 2.72959712e-01,
              7.58221422e-02, 5.87010133e-02, 0.00000000e+00, 1.09183885e-01,
              5.77692512e-02, 6.82399280e-02, 3.63946282e+00, 8.25227036e+00]


# %% Noise now for real


data1 = np.load(r"D:\Session 240312 AFG\3052C\Diff\2024-03-12_15-13-28.npy", allow_pickle=True)


wave_nodepath = "/dev3258/scopes/0/wave"

for record in data1.item()[wave_nodepath]:
    
    plt.plot(record[0]["wave"][0, :])




# %% radom things


folder = r"Z:\Group Members\0 - previous members\Seiler, Toni\barkhausen measurements\15.01.24 Condensator test\10pF 200Hz 18V 938kHz sampling rate"
# folder = r"Z:\temp\Anton F\NOISE BTA für Marcel\Messung Linkam samples vom 04.01.24\last used sample (vlt S3 mm)\square 938kHz 200hz 18.6V"

bhdata = []

for file in os.listdir(folder):
    filename, ext = os.path.splitext(file)

    if not ext == ".txt":
        continue

    t, ch1, ch2, colnames = hf.read_data(os.path.join(folder, file))

    # fig, ax = plt.subplots(1,1)

    # ax.plot(t, ch1)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Current [A]')
    # ax2 = ax.twinx()
    # deriv = np.asarray(hf.calculate_derivative(t, ch1))**2
    # baseline = hf.interpolate_baseline(t, deriv)
    # ax2.plot(t, deriv-baseline, color="red")
    # ax2.set_ylabel(r'Slew-Rate $S=\left[\frac{A^2}{s^2}\right]$')

    bhdata.extend(bhm.analyzeBH(t, ch1, 1E-5))

    print("next")


fig, ax = plt.subplots(1, 1)


edges, hist = powerlaw.pdf(bhdata, number_of_bins=50)
bin_centers = (edges[1:] + edges[:-1]) / 2.0
fit = powerlaw.Fit(bhdata)
fit.power_law.plot_pdf(
    label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha, 2), round(fit.sigma, 2)), ax=ax)
ax.scatter(bin_centers, hist)


histB = [4.58213760e-02, 4.58213760e-02, 3.05475840e-02, 3.05475840e-02,
         0.00000000e+00, 2.54563200e-02, 7.63689600e-03, 2.54563200e-02,
         1.83285504e-02, 3.97118592e-02, 3.56388480e-02, 3.05475840e-02,
         3.66571008e-02, 2.58479557e-02, 3.32429591e-02, 4.29120823e-02,
         4.40590154e-02, 4.76727447e-02, 5.28994747e-02, 5.36023644e-02,
         5.43839564e-02, 5.83348442e-02, 6.05133093e-02, 6.21365629e-02,
         6.55116862e-02, 6.15336501e-02, 6.72856822e-02, 6.74439129e-02,
         6.54695237e-02, 6.23703992e-02, 5.85878738e-02, 5.28188190e-02,
         4.79391098e-02, 4.07684787e-02, 3.52823681e-02, 2.90448868e-02,
         2.27176771e-02, 1.73682222e-02, 1.25175371e-02, 7.93672129e-03,
         4.93485622e-03, 2.79637948e-03, 1.64162970e-03, 9.48312821e-04,
         5.46466781e-04, 2.87313036e-04, 1.73643966e-04, 1.07379276e-04,
         5.56277544e-05, 1.92141206e-05, 1.04730411e-05, 6.50717766e-06,
         3.15704671e-06, 9.11426070e-07, 4.52243245e-07, 1.43616141e-07,
         1.14018409e-07, 9.05204791e-08, 3.59325004e-08, 2.85271753e-08]


bin_centersB = [1.88896786e-03, 3.14827977e-03, 5.03724764e-03, 6.92621550e-03,
                8.18552741e-03, 1.07041512e-02, 1.38524310e-02, 1.70007108e-02,
                2.20379584e-02, 2.83345180e-02, 3.52607335e-02, 4.47055728e-02,
                5.66690359e-02, 7.11511229e-02, 9.00408015e-02, 1.13967728e-01,
                1.43561558e-01, 1.80711259e-01, 2.27305800e-01, 2.86493459e-01,
                3.61422518e-01, 4.55241255e-01, 5.73616575e-01, 7.22845036e-01,
                9.10482510e-01, 1.14660349e+00, 1.44443076e+00, 1.81970571e+00,
                2.29257733e+00, 2.88823186e+00, 3.63815211e+00, 4.58263604e+00,
                5.77205614e+00, 7.27063731e+00, 9.15834586e+00, 1.15352971e+01,
                1.45293112e+01, 1.83015800e+01, 2.30529638e+01, 2.90372140e+01,
                3.65748254e+01, 4.60694076e+01, 5.80290928e+01, 7.30929818e+01,
                9.20670344e+01, 1.15966256e+02, 1.46070107e+02, 1.83988618e+02,
                2.31749911e+02, 2.91909760e+02, 3.67686335e+02, 4.63133363e+02,
                5.83357352e+02, 7.34790239e+02, 9.25533177e+02, 1.16579100e+03,
                1.46841687e+03, 1.84960051e+03, 2.32973522e+03, 2.93450780e+03]
