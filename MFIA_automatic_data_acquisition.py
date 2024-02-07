# Shamelessly adapted from:
# https://github.com/zhinst/labone-api-examples/blob/release-24.1/common/python/example_scope_dig_stream.py
# https://github.com/zhinst/labone-api-examples/blob/release-24.1/common/python/example_scope_dig_dualchannel.py

import time
import warnings
import numpy as np
import zhinst.utils
import matplotlib.pyplot as plt
from matplotlib import cm



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

device_id = "dev3258"
server_host = "192.168.50.234"
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

# daq.setDouble('/dev3258/currins/0/range', 0.00000001)  # <- !10nA! CurrenInput Scale Maybe use auto lateron

daq.sync()


scopeModule = daq.scopeModule()
scopeModule.set("mode", 1) # TODO: Check this!
scopeModule.set("averager/weight", 1)
scopeModule.set("historylength", 50)

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

data_with_trig = get_scope_records(device, daq, scopeModule, 50)

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
    plot_scope_records(axis1, data_with_trig[wave_nodepath], 1)
    axis1.axvline(0.0, linewidth=2, linestyle="--", color="k", label="Trigger time")
    axis1.set_title(f"{len(data_with_trig[wave_nodepath])} Scope records from 2 channels ({device}, triggering enabled)")
    axis1.set_xlabel("t (relative to trigger) [us]")
    fig1.show()


print("end")
