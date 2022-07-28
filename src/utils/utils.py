from scipy.ndimage import zoom
import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict


def create_features_mapping(features_ids, features, all_ids=None, default_value=10):
    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value

    if all_ids is not None:
        for idx in all_ids:
            mapping.setdefault(idx, default_value)
    return mapping


def scale_image(image, in_shape, out_shape, order=0):
    zoom_factor = [i / o for i, o in zip(in_shape, out_shape)]
    return zoom(image, zoom=zoom_factor, order=order)


def scale_image_voxel_size(image, current_voxel_size, out_voxel_size, order=0):
    zoom_factor = [i / o for i, o in zip(current_voxel_size, out_voxel_size)]
    return zoom(image, zoom=zoom_factor, order=order)


@njit(parallel=True)
def _mapping2image(in_image, out_image, mappings):
    shape = in_image.shape
    for i in prange(0, shape[0]):
        for j in prange(0, shape[1]):
            for k in prange(0, shape[2]):
                out_image[i, j, k] = mappings[in_image[i, j, k]]

    return out_image


def create_cell_mapping(features_ids, features):
    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value
    return mapping


def mapping2image(in_image, mappings, data_type='int64'):
    value_type = types.int64 if data_type == 'int64' else types.float64

    numba_mappings = Dict.empty(key_type=types.int64,
                                value_type=value_type)
    numba_mappings.update(mappings)
    numba_mappings[0] = 0

    out_image = np.zeros_like(in_image).astype(data_type)
    out_image = _mapping2image(in_image, out_image, numba_mappings)
    return out_image


def map_cell_features2segmentation(segmentation, cell_ids, cell_feature, data_type='int64'):
    cell_feature_mapping = create_cell_mapping(cell_ids, cell_feature)
    features_image = mapping2image(segmentation, cell_feature_mapping, data_type=data_type)
    return features_image
