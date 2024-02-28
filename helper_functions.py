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
    """
    Reads data from a CSV file generated from MFLI or MFIA, parsing column names and extracting relevant data.

    Parameters
    ----------
    filepath : str
        The path to the CSV file.

    Returns
    -------
    tuple
        A tuple containing:
        - t (array): Array of time values.
        - ch1 (array): Array of values from channel 1.
        - ch2 (array): Array of values from channel 2. If channel 2 is not present,
                        it will be an array of zeros.
        - colnames (list): A list of column names parsed from the CSV file.

    Example
    -------
    >>> t, ch1, ch2, colnames = read_data(meas_scope_20240216_162043.txt)
    """
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
    """
    Slices elements from array1 based on the positions marked as 1 in array2.

    Parameters
    ----------
        array1 : list
            The original list from which elements will be sliced.
        array2 : list
            A binary indicator list where 1 represents the positions
            where elements should be sliced from array1.

    Returns
    ----------
        list 
            A list containing slices of elements from array1 based on the positions
            marked as 1 in array2. Each slice is represented as a sublist.

    Example
    ----------
    
    >>> array1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> array2 = [0, 1, 1, 0, 0, 1, 1, 0, 0]
    >>> sliced_arrays = slice_array(array1, array2)
    [[2, 3], [6, 7]]
    """
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
    """
    Replaces 0s in an array with 1s, infront of a 1 bock, based on the specified number.

    Parameters
    ----------
    - arr (list): The original list where 0s may be replaced with 1s.
    - num_ones (int): The number of preceding 1s to be added

    Returns
    -------
    - list: A modified list where 0s have been replaced with 1s based on the specified number

    Example
    --------
    >>> preplace_ones([0, 0, 1, 1, 0, 0, 1, 1, 0], 1)
                      [0, 1, 1, 1, 0, 1, 1, 1, 0]
    """
    
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
    """
    Reduces consecutive sequences of ones in an array by replacing the last 'n' ones with zeros.
    
    Parameters
    ----------
    array : list
        The input list containing elements, including ones and zeros.
    n : int
        The number of consecutive ones to be replaced with zeros from the end of each sequence.
    
    Returns
    -------
    list
        A modified list where the last 'n' ones in each consecutive sequence of ones have been replaced with zeros.
    
    Examples
    --------
    >>> reduce_ones([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 2)
                    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    >>> reduce_ones([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], 1)
                    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]

    """
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
    Pads arrays to have a length of 4096 elements.

    Parameters
    ----------
    arrays : List[np.ndarray]
        A list of arrays where each array represents a sequence of data.

    Returns
    -------
    List[np.ndarray]
        A list of padded arrays where each array has a length of 4096 elements.
        If an array is shorter than 4096, it's padded with zeros.
        If the timestamp array (first array) is shorter than 4096, it's linearly interpolated.

    """
    
    padded_arrays = []

    for sub_array in arrays:
        padded_sub_array = []

        # Pad the timestamp array (first array) to 4096 with linear interpolation
        timestamp_array = sub_array[0]
        timestamp_length = len(timestamp_array)
        if timestamp_length >= 4096:
            padded_sub_array.append(timestamp_array[:4096])
        else:
            # Linearly interpolate to extend the timestamp array
            extended_timestamp = np.linspace(
                timestamp_array[0], timestamp_array[-1], num=4096)
            padded_sub_array.append(extended_timestamp)

        # Pad other arrays with zeros
        for array in sub_array[1:]:
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
    """
    Converts time in different units to the equivalent number of samples based on the given sample rate.

    Parameters
    ----------
    time : float
        The time duration to be converted to samples.
    unit : str
        The unit of the input time. It can be one of the following: 's' (seconds), 'ms' (milliseconds),
        or 'us' (microseconds).
    samplerate : int
        The sample rate in samples per second.

    Returns
    -------
    int
        The equivalent number of samples for the given time duration.

    Raises
    ------
    NotImplementedError
        If the provided unit is not one of 's', 'ms', or 'us'.

    Example
    -------
    >>> time_to_samples(1.5, 's', 44100)
    66150
    >>> time_to_samples(100, 'ms', 48000)
    4800
    >>> time_to_samples(500, 'us', 96000)
    48
    """
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
    """
    Finds the index of the array whose absolute minimum value is closest to zero among a list of arrays.
    
    Parameters
    ----------
    arrays : list of lists
        A list of arrays where each array represents a sequence of numeric values.
    
    Returns
    -------
    int or None
        The index of the array in the list `arrays` whose absolute minimum value is closest to zero.
        If the list `arrays` is empty, returns None.
    
    Example
    -------
    >>> arrays = [[1, 2, 3], [-5, 8, 10], [4, -7, 6]]
    >>> closest_to_zero_index(arrays)
    0
    """
    
    min_abs_val = float('inf')
    closest_index = None

    for i, array in enumerate(arrays):
        min_val = min(abs(x) for x in array)
        if min_val < min_abs_val:
            min_abs_val = min_val
            closest_index = i

    return closest_index


def closest_key(dictionary, value):
    """
    Finds the key in the dictionary whose corresponding value or values are closest to a given value.
    
    Parameters
    ----------
    dictionary : dict
        A dictionary where keys are identifiers and values are either single numbers or lists of numbers.
    value : numeric
        The target value to which the dictionary values should be compared.
    
    Returns
    -------
    any
        The key in the dictionary whose corresponding value or values are closest to the given value.
        If the dictionary is empty or all values are empty lists, returns None.
    
    Example
    -------
    >>> dictionary = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': 7}
    >>> closest_key(dictionary, 4.5)
    'B'
    """
    
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


# TODO: Check if this ever got used

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


def interpolate_baseline(data_x, data_y):
    """
    Interpolates baseline from the data using local minima.

    Parameters
    ----------
    data_x : array-like
        The x-axis values of the data.
    data_y : array-like
        The y-axis values of the data.

    Returns
    -------
    array-like
        The interpolated baseline.

    Notes
    -----
    This function identifies local minima in the data and interpolates a baseline using piecewise cubic Hermite interpolation.
    It removes baseline made from local minima and can be set to either only take negative minima for baseline or everything.
    """
    
    minlist = []
    timelist = []

    # remove baseline made from local minima can be set to either only take negative minima for baseline or everything
    for l in range(0, len(data_y)):

        if l == 0:
            if data_y[l] < data_y[l+1]:
                minlist.append(data_y[l])
                timelist.append(data_x[l])

        elif l == len(data_y)-1:
            if data_y[l] < data_y[l-1]:
                minlist.append(data_y[l])
                timelist.append(data_x[l])

        elif l > 0 and l < len(data_y)-1:
            if data_y[l] < data_y[l-1] and data_y[l] < data_y[l+1]:
                minlist.append(data_y[l])
                timelist.append(data_x[l])

    baseline = scipy.interpolate.pchip_interpolate(timelist, minlist, data_x)

    return baseline



# TODO: This might have a bug! It doesnt matter for [{timestamp:[...],value:[...]}]
# but should be look at see example
def concat_dict_array(existing_array, additional_array):
    """
    Concatenates dictionaries in two arrays by updating the existing array in-place.

    Parameters
    ----------
    existing_array : List[Dict]
        The list of dictionaries to be updated.
    additional_array : List[Dict]
        The list of dictionaries to be concatenated with the existing array.

    Returns
    -------
    None
        The function operates in-place and does not return any value.

    Notes
    -----
    This function iterates through the dictionaries in both arrays and concatenates corresponding values if the keys match.
    If a key exists in both dictionaries, the values are concatenated using NumPy's concatenate function.
    If a key exists in the additional dictionary but not in the existing one, the key-value pair is added to the existing dictionary.

    Example
    -------
    >>> existing_array = [{'A': [1, 2, 3]}, {'B': [4, 5, 6]}]
    >>> additional_array = [{'A': [7, 8, 9]}, {'C': [10, 11, 12]}]
    >>> concat_dict_array(existing_array, additional_array)
    >>> print(existing_array)
    [{'A': array([1, 2, 3, 7, 8, 9])}, {'B': [4, 5, 6], 'C': [10, 11, 12]}]
    """
    for existing_dict, new_dict in zip(existing_array, additional_array):

        for key in new_dict.keys():

            if key in existing_dict.keys():
                existing_dict[key] = np.concatenate((existing_dict[key], new_dict[key]))
            else:
                existing_dict[key] = new_dict[key]


def calculate_derivative(data_x, data_y, dtype="simple"):
    """
    Calculates the derivative of y with respect to x using finite difference methods.

    Parameters
    ----------
    data_x : array-like
        The x-axis values of the data.
    data_y : array-like
        The y-axis values of the data.
    dtype : str, optional
        The type of derivative calculation to perform. It can be either "simple" or "five-point".
        Defaults to "simple".

    Returns
    -------
    array-like
        The calculated derivative values.

    Raises
    ------
    NotImplementedError
        If the provided dtype is not supported.

    Notes
    -----
    This function calculates the derivative of y with respect to x using finite difference methods.
    Two types of derivative calculations are supported:
    - Simple finite difference (dtype="simple"): Uses the formula f'(x) = (f(x+h) - f(x)) / h.
    - Five-point stencil finite difference (dtype="five-point"): Uses a five-point stencil for derivative approximation.

    Example
    -------
    >>> import numpy as np
    >>> data_x = np.array([0, 1, 2, 3, 4, 5])
    >>> data_y = np.array([0, 1, 4, 9, 16, 25])
    >>> derivative = calculate_derivative(data_x, data_y, dtype="simple")
    >>> print(derivative)
    array([1., 3., 5., 7., 7., 7.])
    """
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

                data_yp[i] = (data_y[i-2]-8*data_y[i-1]+8 *
                              data_y[i+1]-data_y[i+2])/(12*dt)

        return data_yp

    else:
        raise NotImplementedError()
        return None


# TODO: this is also somehow broken
# Absolute code removed add later again!
#

def group_numbers_within(numbers, diffrence, absolute=False):
    """
    Groups numbers within a certain difference threshold.

    Parameters
    ----------
    numbers : list of numeric
        The list of numbers to be grouped.
    diffrence : float
        The maximum allowed difference between numbers for them to be grouped together.
    absolute : bool, optional
        If True, the difference is treated as absolute. Defaults to False.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains the starting number of the group and the indices of numbers within the group.

    Notes
    -----
    This function sorts the numbers and groups them based on their differences.
    Numbers within the specified difference (or absolute difference) are grouped together.
    If the difference is not absolute, each number is compared with the previous number in the group multiplied by (1 + diffrence).
    If absolute is True, each number is compared with the previous number plus the specified difference.

    Example
    -------
    >>> numbers = [1, 2, 4, 6, 7, 9, 11, 14, 16]
    >>> group_numbers_within(numbers, 1)
    [(1, [0, 1]), (4, [2]), (6, [3, 4]), (9, [5]), (11, [6]), (14, [7]), (16, [8])]
    """
    
    numbers_with_index = [(num, i) for i, num in enumerate(numbers)]
    numbers_with_index.sort(key=lambda x: x[0])

    groups = []
    current_group = []

    for num, index in numbers_with_index:

        maxval = current_group[-1][0] * (1+diffrence)

        if not current_group or num <= maxval:
            current_group.append((num, index))
        else:
            groups.append((current_group[0][0], [x[1] for x in current_group]))
            current_group = [(num, index)]

    # Add the last group
    if current_group:
        groups.append((current_group[0][0], [x[1] for x in current_group]))

    return groups


def extract_consecutive(input_list):
    values = []
    indices = []

    if len(input_list) == 0:
        return values, indices

    current_value = input_list[0]
    current_index = 0

    for index, value in enumerate(input_list[1:], start=1):
        if value != current_value:
            values.append(current_value)
            indices.append(current_index)
            current_value = value
            current_index = index

    values.append(current_value)
    indices.append(current_index)

    values = np.asarray(values)
    indices = np.asarray(indices)

    return values, indices


def calculate_differences_every_second(arr):
    differences = []
    for i in range(len(arr) - 2):
        if i + 2 < len(arr):
            diff = arr[i + 2] - arr[i]
            differences.append(diff)
    return differences

if __name__ == "__main__":
    print("This is just a file with functions nothing more")


# Yes, i know a SchmittTrigger could also be implemented as a FSM.
# like
#  transition =     {
#                       0: {1: 1, 0: 0}.
#                       1: {1: 1, 0: 0}
#                   }
#   fsm.process_input(1 if x >= high elif 0 x =< low)
#

class SchmittTrigger:
    """
    Implements a Schmitt trigger.

    Attributes
    ----------
    v_high : float
        The upper threshold voltage. When the input voltage exceeds this threshold, the output is set to 1.
    v_low : float
        The lower threshold voltage. When the input voltage falls below this threshold, the output is set to 0.
    output : int
        The current output state of the Schmitt trigger.

    Methods
    -------
    __init__(v_high, v_low)
        Initializes the SchmittTrigger object with the specified high and low voltage thresholds.
    process_input(input_voltage)
        Processes the input voltage and determines the output state of the Schmitt trigger.
    """
    
    def __init__(self, v_high, v_low):
        
        """
        Initializes a SchmittTrigger object.

        Parameters
        ----------
        v_high : float
            The upper threshold voltage.
        v_low : float
            The lower threshold voltage.
        """
        
        self.v_high = v_high
        self.v_low = v_low
        self.output = 0
        
    
    def reset(self):
        self.output = 0
    
    def process_input(self, input_voltage):
        
        """
        Processes the input voltage and determines the output state of the Schmitt trigger.

        Parameters
        ----------
        input_voltage : float
            The input voltage to be processed.

        Returns
        -------
        int
            The output state of the Schmitt trigger (0 or 1).
        """
        
        if input_voltage >= self.v_high:
            self.output = 1
        elif input_voltage <= self.v_low:
            self.output = 0

        return self.output


class StateMachine:
    def __init__(self, init_state, transitions, return_state=False):
        
        self.transitions = transitions
        self.current_state = init_state
        self.init_state = init_state
        self.return_state = return_state

    def process_input(self, input_sequence):
        for inp in input_sequence:
            if inp in self.transitions[self.current_state]:
                self.current_state = self.transitions[self.current_state][inp]
            else:
                print(f"Invalid transition from {self.current_state} with input {inp}")
                raise NotImplementedError()
                break
            
            if self.return_state:
                return self.current_state
            
            
    def reset(self):
        self.current_state = self.init_state
        
            
    def __str__(self):
        return self.current_state