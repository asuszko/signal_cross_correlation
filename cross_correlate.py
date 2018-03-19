# -*- coding: utf-8 -*-

__all__ = [
    "device_obj",
    "run",
]

import numpy as np
from pycu_interface import Device
import warnings

# Custom written C/C++ & CUDA from shared library.
from lib_import import (copy_signal_mf, cross_correlate)


dtype_map = {np.dtype('f4'):0,
             np.dtype('f8'):1}


class device_obj(Device, object):
    """
    Object that initializes the device. Making this as an 
    object is useful to store device pointer references 
    within, which can then be passed around.
    """    
    def __init__(self, batch_size, dtype, device_id=0, n_streams=1):
        """
        Perform one time memory allocations, and other device 
        initializtion operations. Here, the data is chunked up 
        among the streams in j-dimension.
        """
        # Inherit the Device object
        super(device_obj, self).__init__(device_id, n_streams)

        batch_y, signal_len  = batch_size
        batch_y_s = batch_y//n_streams
        
        # Allocating memory chunk for each stream
        for s in self.streams:
            s.data = s.malloc((signal_len, batch_y_s), dtype=dtype)
            s.corrcoefs = s.malloc(batch_y_s, dtype=dtype)

    

def run(device_obj, ref_signal, data, corr_coefs):
    """
    This function runs a CUDA accelerate normalized 
    cross correlation between a reference signal and 
    a batch of signals.
    
    Parameters
    ----------
    dev_obj : Device
        The initialized device object.

    ref_signal : np.ndarray (1d)
        The reference signal to cross correlate against.

    data : np.ndarray (2d)
        The batch of signals to correlate against ref_signal.
        Each individual signal is in its own row.
        
    Notes
    -----
    Instead of returning corr_coeffs, this example passes 
    in a reference to be filled. It can be initialized 
    within this function, and returned.
    """
    
    # Signals must be C_CONTIGUOUS
    if not data.flags['C_CONTIGUOUS']:
        data = np.require(data, requirements='C')
        warnings.warn('data not C_CONTIGUOUS. Making C_CONTIGUOUS...')

  
    # Image dimenions must match
    if len(ref_signal) != data.shape[1]:
        raise ValueError('Error: Reference signal and data dimensions mismatch.')

      
    # Make sure the signals are device streamable since we are using streams.
    device_obj.require_streamable(ref_signal, data, corr_coefs)    

   
    # Copy the reference signal to device constant memory as needed by the kernel.
    copy_signal_mf(ref_signal, ref_signal.nbytes)
    
    
    batch_y, signal_len = data.shape
    batch_y_s = batch_y//len(device_obj.streams)
    dims = np.array([signal_len,batch_y_s], dtype='i4')
    
    # Run the normalized cross correlation on the GPU stream(s)
    for stream_id, s in enumerate(device_obj.streams):
        
        data_chunk_start = stream_id*batch_y_s
        data_chunk_end = (stream_id+1)*batch_y_s
        s.data.h2d_async(data[data_chunk_start:data_chunk_end], s.stream)
        
        cross_correlate(s.corrcoefs.ptr,
                        s.data.ptr,
                        dims,
                        dtype_map[data.dtype],
                        s.stream)
                        
        s.corrcoefs.d2h_async(corr_coefs[data_chunk_start:data_chunk_end], s.stream)