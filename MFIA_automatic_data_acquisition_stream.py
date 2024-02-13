# Shamelessly adapted from:
# https://github.com/zhinst/labone-api-examples/blob/release-24.1/common/python/example_scope_dig_stream.py
# https://github.com/zhinst/labone-api-examples/blob/release-24.1/common/python/example_scope_dig_dualchannel.py

import time
import warnings
import numpy as np
import zhinst.utils
import matplotlib.pyplot as plt
import queue
import threading
import helper_functions as hf
from datetime import datetime


def aquire_data(num_scope_samples):

    num_scope = 0  # The number of scope samples acquired on each channel.
    num_blocks = 0  # Just for statistics.
    poll_count = 0
    timeout = 60
    t_start = time.time()

    scope_samples = [
        {
            "value": np.nan * np.ones(num_scope_samples),
            "timestamp": np.zeros(num_scope_samples, dtype=int),
        },
        {
            "value": np.nan * np.ones(num_scope_samples),
            "timestamp": np.zeros(num_scope_samples, dtype=int),
        },
    ]

    while num_scope < num_scope_samples:

        if time.time() - t_start > timeout:  # This logic errors if it takes longer than timeout to aquire all samples
            raise Exception(
                "Failed to acquired %d scope samples after %f s. Num samples acquired"
            )

        data = daq.poll(0.02, 200, 0, True)
        poll_count += 1

        if stream_nodepath not in data:
            # Could be the case for very slow streaming rates and fast poll frequencies.
            print("Poll did not return any subscribed data.")
            continue

        num_blocks_poll = len(data[stream_nodepath])
        num_blocks += num_blocks_poll

        print(
            f"Poll #{poll_count} returned {num_blocks_poll} blocks of streamed scope data. "
            f"blocks processed {num_blocks}, samples acquired {num_scope}.\r"
        )

        for num_block, block in enumerate(data[stream_nodepath]):

            if block["flags"] & 1:
                message = f"Block {num_block} from poll indicates dataloss \
                    (flags: {block['flags']})"
                warnings.warn(message)
                continue

            if block["flags"] & 2:
                # This should not happen.
                message = f"Block {num_block} from poll indicates missed trigger \
                    (flags: {block['flags']})"
                warnings.warn(message)
                continue

            if block["flags"] & 3:
                message = f"Block {num_block} from poll indicates transfer failure \
                    (flags: {block['flags']})"
                warnings.warn(message)

            assert (
                block["datatransfermode"] == 3
            ), "The block's datatransfermode states the block does not contain scope streaming \
                data."

            # The same for all channels.
            num_samples_block = len(block["wave"][:, 0])

            if num_samples_block + num_scope > num_scope_samples:
                num_samples_block = num_scope_samples - num_scope

            # The delta inbetween timestamps.
            ts_delta = int(clockbase * block["dt"])

            for (i,), channelenable in np.ndenumerate(block["channelenable"]):

                if not channelenable:
                    continue
                # 'timestamp' is the last sample's timestamp in the block.
                ts_end = (
                    block["timestamp"]
                    - (len(block["wave"][:, i]) - num_samples_block) * ts_delta
                )

                ts_start = ts_end - num_samples_block * ts_delta
                scope_samples[i]["timestamp"][
                    num_scope: num_scope + num_samples_block
                ] = np.arange(ts_start, ts_end, ts_delta)

                scope_samples[i]["value"][num_scope: num_scope + num_samples_block] = (
                    block["channeloffset"][i]
                    + block["channelscaling"][i] *
                    block["wave"][:num_samples_block, i]
                )

            num_scope += num_samples_block

    # print()
    # print(f"Total blocks processed {num_blocks}, samples acquired {num_scope}.")

    return scope_samples


def aquire_data_thread():

    num_scope = 0
    num_blocks = 0
    poll_count = 0
    i = 0

    while i < 5000:
        i += 1

        data = daq.poll(0.02, 200, 0, True)
        poll_count += 1

        if stream_nodepath not in data:
            print("Poll did not return any subscribed data.")
            continue

        num_blocks_poll = len(data[stream_nodepath])
        num_blocks += num_blocks_poll

        print(
            f"Poll #{poll_count} returned {num_blocks_poll} blocks of streamed scope data. "
            f"blocks processed {num_blocks}, samples acquired {num_scope}.\r"
        )

        poll_data = [{}, {}]

        for num_block, block in enumerate(data[stream_nodepath]):
            block_data = process_block(num_block, block)

            hf.concat_dict_array(poll_data, block_data)

        dataqueue.put(poll_data)


def process_block(num_block, block):

    block_data = [{}, {}]

    if block["flags"] & 1:
        message = f"Block {num_block} from poll indicates dataloss (flags: {block['flags']})"
        warnings.warn(message)
        return None

    if block["flags"] & 2:
        # This should not happen.
        message = f"Block {num_block} from poll indicates missed trigger (flags: {block['flags']})"
        warnings.warn(message)
        return None

    if block["flags"] & 3:
        message = f"Block {num_block} from poll indicates transfer failure (flags: {block['flags']})"
        warnings.warn(message)

    assert (block["datatransfermode"] ==
            3), "The block's datatransfermode states the block does not contain scope streaming data."

    num_samples_block = len(block["wave"][:, 0])

    ts_delta = int(clockbase * block["dt"])

    for (i,), channelenable in np.ndenumerate(block["channelenable"]):

        if not channelenable:
            continue

        ts_end = (block["timestamp"] - (len(block["wave"]
                  [:, i]) - num_samples_block) * ts_delta)
        ts_start = ts_end - num_samples_block * ts_delta
        block_data[i]["timestamp"] = np.arange(ts_start, ts_end, ts_delta)
        block_data[i]["value"] = block["channeloffset"][i] + \
            block["channelscaling"][i] * block["wave"][:, i]

    return block_data


def process_data_thread():

    queuehasdata = True
    filenumber = 1
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M")

    while queuehasdata:

        chunk = [{}, {}]

        for _ in range(100):

            try:

                hf.concat_dict_array(chunk, dataqueue.get(timeout=1))

            except queue.Empty:
                print("Queue is empty.")
                queuehasdata = False
                break

        if len(chunk[0]) > 0:
            np.savez_compressed(starttime + "_" + str(filenumber), chunk)
            filenumber += 1


input1 = 1  # "currin0", "current_input0": Current Input 1
input2 = 8  # "auxin0", "auxiliary_input0": Aux Input 1
stream_rate = 6  # "938_kHz": 938 kHz


device_id = "dev3258"
server_host = "192.168.50.234"
server_port = 8004

apilevel = 6


(daq, device, _) = zhinst.utils.create_api_session(device_id, apilevel, server_host=server_host, server_port=server_port)

zhinst.utils.api_server_version_check(daq)

daq.setDebugLevel(3)

zhinst.utils.disable_everything(daq, device)

clockbase = daq.getInt(f"/{device}/clockbase")
rate = clockbase / 2**stream_rate

daq.sync()

daq.setInt("/%s/scopes/0/channels/0/inputselect" % device, input1)
daq.setInt("/%s/scopes/0/channels/1/inputselect" % device, input2)
daq.setInt("/%s/scopes/0/channels/*/bwlimit" % device, 1)

daq.sync()

daq.setDouble("/%s/scopes/0/stream/rate" % device, stream_rate)
daq.sync()

daq.setInt("/%s/scopes/0/stream/enables/0" % device, 1)
daq.setInt("/%s/scopes/0/stream/enables/1" % device, 1)
daq.sync()

# Subscribe to the scope's streaming samples in the ziDAQServer session.
stream_nodepath = f"/{device}/scopes/0/stream/sample"
daq.subscribe(stream_nodepath)


dataqueue = queue.Queue()


dataaquithread = threading.Thread(target=aquire_data_thread)
dataevalthread = threading.Thread(target=process_data_thread)

dataaquithread.start()
dataevalthread.start()


dataaquithread.join()


daq.sync()
daq.setInt("/%s/scopes/0/stream/enables/*" % device, 0)
daq.unsubscribe("*")

dataevalthread.join()


# expected_ts_delta = 2**stream_rate

# for num_scope, channel_samples in enumerate(scope_samples):
#     # Check for sampleloss
#     nan_count = np.sum(np.isnan(scope_samples[num_scope]["value"]))
#     zero_count = np.sum(scope_samples[num_scope]["timestamp"] == 0)
#     diff_timestamps = np.diff(scope_samples[num_scope]["timestamp"])
#     min_ts_delta = np.min(diff_timestamps)
#     max_ts_delta = np.max(diff_timestamps)
#     if nan_count:
#         nan_index = np.where(np.isnan(scope_samples[num_scope]["value"]))[0]
#         warnings.warn(
#             "Scope channel %d values contain %d/%d nan entries (starting at index %d)."
#             % (
#                 num_scope,
#                 int(nan_count),
#                 len(scope_samples[num_scope]["value"]),
#                 nan_index[0],
#             )
#         )
#     if zero_count:
#         warnings.warn(
#             "Scope channel %d timestamps contain %d entries equal to 0."
#             % (num_scope, int(zero_count))
#         )
#     ts_delta_mismatch = False
#     if min_ts_delta != expected_ts_delta:
#         index = np.where(diff_timestamps == min_ts_delta)[0]
#         warnings.warn(
#             "Scope channel %d timestamps have a min_diff %d (first discrepancy at pos: %d). "
#             "Expected %d." % (num_scope, min_ts_delta, index[0], expected_ts_delta)
#         )
#         ts_delta_mismatch = True
#     if max_ts_delta != expected_ts_delta:
#         index = np.where(diff_timestamps == max_ts_delta)[0]
#         warnings.warn(
#             "Scope channel %d timestamps have a max_diff %d (first discrepenacy at pos: %d). "
#             "Expected %d." % (num_scope, max_ts_delta, index[0], expected_ts_delta)
#         )
#         ts_delta_mismatch = True
#     dt = (
#         channel_samples["timestamp"][-1] - channel_samples["timestamp"][0]
#     ) / float(clockbase)
#     print(
#         "Samples in channel",
#         num_scope,
#         "span",
#         dt,
#         "s at a rate of",
#         rate / 1e3,
#         "kHz.",
#     )
#     assert not nan_count, "Detected NAN in the array of scope samples."
#     assert (
#         not ts_delta_mismatch
#     ), "Detected an unexpected timestamp delta in the scope data."


# for _, channel_samples in enumerate(scope_samples):
#
#            #t = (channel_samples["timestamp"] - channel_samples["timestamp"][0]) / clockbase
#            plt.plot(channel_samples["timestamp"], channel_samples["value"])
#            plt.show()


print("Exit")
