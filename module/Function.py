import numpy as np


def ReLU(inputs):
    return np.maximum(0, inputs)

def Softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)
    slided_inputs = inputs - max_values
    exp_value = np.exp(slided_inputs)
    norm_base = np.sum(exp_value, axis=1, keepdims=True)
    norm_values = exp_value/norm_base
    return norm_values

def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 0, 1/max_number)
    norm = array * scale_rate
    return norm

def classfify(probabilities):
    classification = np.rint(probabilities[:,1])
    return classification