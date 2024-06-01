from __future__ import absolute_import

import numpy as np
from unwrap.unwrap2D import unwrap2D


def unwrap(wrapped_array,
           wrap_around_axis_0 = False,
           wrap_around_axis_1 = False,
           quality_array=None):

    wrapped_array = np.require(wrapped_array, np.float32, ['C'])
    if quality_array is None:
        quality_array=np.ones_like(wrapped_array)

    quality_array = np.require(quality_array, np.float32, ['C'])
    
    if wrapped_array.ndim != quality_array.ndim:
        raise ValueError('mismatch between wrapped and quality')
    if wrapped_array.ndim != 2:
        raise ValueError('input array needs to have 2 dimensions')

    wrapped_array_masked = np.ma.asarray(wrapped_array)
    quality_array_masked = np.ma.asarray(quality_array)
    unwrapped_array = np.empty_like(wrapped_array_masked.data)
    unwrap2D(wrapped_array_masked.data, quality_array_masked.data,
             np.ma.getmaskarray(wrapped_array_masked).astype(np.uint8),
             unwrapped_array,
             bool(wrap_around_axis_0), bool(wrap_around_axis_1))

    if np.ma.isMaskedArray(wrapped_array):
        return np.ma.array(unwrapped_array, mask = wrapped_array_masked.mask)
    else:
        return unwrapped_array

    #TODO: set_fill to minimum value
    #TODO: check for empty mask, not a single contiguous pixel

