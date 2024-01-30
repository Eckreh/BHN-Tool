# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:45:11 2024

@author: av179
"""

import pandas as pd
import numpy as np
import copy


def read_data(filepath):

    colnames = []

    with open(filepath, 'r') as file:

        last_line = ""

        for line_number, line in enumerate(file, start=1):
            if line.startswith("%"):
                last_line = line
            else:
                colnames = last_line.replace(
                    '%', "").replace("\n", "").split(";")
                break

    data = pd.read_csv(filepath, comment='%', sep=';',
                       names=colnames).values

    t = data[0:, 0]
    ch1 = data[0:, 1]
    ch2 = np.zeros(len(t))

    if len(colnames) > 2:
        # TODO: previous check was np.isnan() maybe diffrent cases have to see
        if not data[0, 2] == " ":
            ch2 = data[0:, 2]

    return t, ch1, ch2, colnames


def slice_array(array1, array2):

    slices = []
    current_slice = []

    for index, value in enumerate(array2):
        if value == 1:
            current_slice.append(array1[index])
        elif current_slice:
            slices.append(current_slice)
            current_slice = []

    if current_slice:
        slices.append(current_slice)

    return slices


def preplace_ones(arr, num_ones):

    arr_copy = copy.deepcopy(arr)
    for i in range(len(arr_copy)):
        if arr_copy[i] == 1:
            for j in range(1, num_ones + 1):
                if i - j >= 0 and arr[i - j] == 0:
                    arr_copy[i - j] = 1
                else:
                    break
    return arr_copy


def time_to_samples(time, unit, samplerate):

    samplerate = int(samplerate)

    if unit == "s":
        return int(samplerate * time)
    elif unit == "ms":
        return int(samplerate * (time  / 1000))
    elif unit == "us":
        return int(samplerate * (time / 1E6))
    else:
        raise NotImplementedError()
        return None

def closest_to_zero_index(arrays):
    min_abs_val = float('inf')
    closest_index = None

    for i, array in enumerate(arrays):
        min_val = min(abs(x) for x in array)
        if min_val < min_abs_val:
            min_abs_val = min_val
            closest_index = i

    return closest_index


class SchmittTrigger:

    def __init__(self, v_high, v_low):
        self.v_high = v_high
        self.v_low = v_low
        self.output = 0

    def process_input(self, input_voltage):
        if input_voltage >= self.v_high:
            self.output = 1
        elif input_voltage <= self.v_low:
            self.output = 0

        return self.output
