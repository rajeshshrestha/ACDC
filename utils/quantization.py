import numpy as np


def is_vector(x):
    x_shape = np.shape(x)

    if len(x_shape) == 0 or len(x_shape) > 2:
        return False

    if len(x_shape) == 1:
        return True

    if x_shape[0] == 1 or x_shape[1] == 1:
        return True
    else:
        return False


def quantize(x, partition, codebook, return_distortion=False):
    '''Check Valid arguments'''
    assert np.size(x) != 0 and np.all(np.isreal(x)) , "Invalid signal value passed!!!"
    assert np.size(partition) != 0 and np.all(np.isreal(partition)) and is_vector(
        partition) and np.all(np.sort(partition) == partition), "Invalid partition passed!!!"
    assert np.size(codebook) != 0 and is_vector(codebook) and np.all(np.isreal(
        codebook)) and np.size(codebook) == np.size(partition)+1, "Invalid codebook passed!!!"

    idxs = np.digitize(x, bins=partition, right=True)
    quantized_x = codebook[idxs]
    
    if return_distortion:
        distortion = np.linalg.norm(x-quantized_x)**2
        # * Normalized distortion per number of signal
        distortion /= np.size(x)
    else:
        distortion=None

    return idxs, quantized_x, distortion


def quantize_uniform(x, n_bits, return_distortion=False, original_scale=True, min_value=None, max_value=None):
    
    '''Normalize signal to have max value of 2**n_bits'''
    if min_value is None:
        min_value = np.min(x) 
    if max_value is None:
        max_value = np.max(x)

    scale_ratio = 2**(n_bits+1)/(max_value-min_value)
    x_shift = (max_value + min_value)/2
    x_normalised = (x-x_shift)*scale_ratio

    '''Generate partition and codebook'''
    partition = np.arange(-2**n_bits+2, 2**n_bits-2+1, 2)
    codebook = np.arange(-2**n_bits+1, 2**n_bits-1+1, 2)

    idxs, quantized_x, distortion =  quantize(x_normalised, partition, codebook, return_distortion)

    b_up, b_low = quantized_x+1, quantized_x-1 # boundaries for each reading
    
    '''Scale back to original signal scale'''
    if original_scale:
        quantized_x = (quantized_x / scale_ratio) + x_shift
        b_up = (b_up / scale_ratio) + x_shift
        b_low = (b_low / scale_ratio) + x_shift

    return idxs, quantized_x, b_up, b_low, scale_ratio, distortion