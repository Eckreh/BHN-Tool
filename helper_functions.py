# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:45:11 2024

@author: av179
"""

import pandas as pd
import scipy
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


def reduce_ones(array, n):
    arr_copy = copy.deepcopy(array)

    i = 0
    while i < len(arr_copy):
        if arr_copy[i] == 1:
            j = i
            while j < len(arr_copy) and arr_copy[j] == 1:
                j += 1
                
            for k in range(j - n, j):
                if k >= 0:
                    arr_copy[k] = 0
            i = j
        else:
            i += 1
            
    return arr_copy


def pad_arrays_to_4096(arrays):
    """
    Pad each array within a multidimensional array to a length of 4096 using np.pad.
    
    Parameters:
    - arrays: The input multidimensional array containing arrays to be padded.
    
    Returns:
    - padded_arrays: The padded multidimensional array.
    """
    padded_arrays = []
    
    for sub_array in arrays:
        padded_sub_array = []
        
        for array in sub_array:
            length = len(array)
            if length >= 4096:
                padded_sub_array.append(array[:4096])
            else:
                pad_width = (0, 4096 - length)
                padded_array = np.pad(array, pad_width, mode='constant')
                padded_sub_array.append(padded_array)
                
        padded_arrays.append(padded_sub_array)
        
    return padded_arrays


def time_to_samples(time, unit, samplerate):

    samplerate = int(samplerate)

    if unit == "s":
        return int(samplerate * time)
    elif unit == "ms":
        return int(samplerate * (time / 1000))
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


def closest_key(dictionary, value):
    closest_key = None
    min_difference = float('inf')  # Initialize with infinity

    for key, nums in dictionary.items():
        if isinstance(nums, list):
            for num in nums:
                difference = abs(num - value)
                if difference < min_difference:
                    min_difference = difference
                    closest_key = key

    return closest_key


def combine_points(array):
    combined_array = []
    i = 0
    while i < len(array):
        # If there's an odd number of elements left, just append the last one
        if i == len(array) - 1:
            combined_array.append(array[i])
        else:
            # Combine two adjacent points (e.g., by averaging)
            combined_value = (array[i] + array[i + 1]) / 2.0
            combined_array.append(combined_value)
        i += 2  # Move to the next pair of points
    return combined_array


def interpolate_baseline(data_x, data_y, threshold=0):
    minlist=[]
    timelist=[]
    
    #remove baseline made from local minima can be set to either only take negative minima for baseline or everything 
    for l in range(0, len(data_y)):
        
         if l == 0:
             if data_y[l]<data_y[l+1]:
                 minlist.append(data_y[l])
                 timelist.append(data_x[l])
     
         elif l == len(data_y)-1:
             if data_y[l]<data_y[l-1]:
                 minlist.append(data_y[l])
                 timelist.append(data_x[l])
     
         elif l > 0 and l < len(data_y)-1:
             if data_y[l] < data_y[l-1] and data_y[l] < data_y[l+1]:
                 minlist.append(data_y[l])
                 timelist.append(data_x[l])
    
    baseline = scipy.interpolate.pchip_interpolate(timelist, minlist, data_x)
    
    return baseline


def concat_dict_array(existing_array, additional_array):
    
    for existing_dict, new_dict in zip(existing_array, additional_array):
        
        for key in new_dict.keys():
        
            if key in existing_dict.keys():
                existing_dict[key] = np.concatenate((existing_dict[key], new_dict[key]))
            else:
                existing_dict[key] = new_dict[key]


def calculate_derivative(data_x, data_y, dtype="simple"):
    
    
    if dtype == "simple":
        
        # f'(x) = (f(x+h)-f(x))/h
        
        data_yp = np.zeros_like(data_y)
        
        
        for i, f in enumerate(data_y):
            if i > 0 and i < len(data_y)-1:
                dt = data_x[i+1] - data_x[i]
                data_yp[i] = (data_y[i+1] - f) / dt
        
        return data_yp
        
    elif dtype == "five-point":
       
        data_yp = np.zeros_like(data_y)
        
        for i, f in enumerate(data_y):
            if i > 4 and i < len(data_y)-4:
                dt = data_x[i+1]-data_x[i]
                
                data_yp[i] = (data_y[i-2]-8*data_y[i-1]+8*data_y[i+1]-data_y[i+2])/(12*dt)
        
        return data_yp
    
    else:
        raise NotImplementedError()
        return None


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
