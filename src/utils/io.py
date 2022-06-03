import csv
from pathlib import Path

import h5py
import numpy as np
import tifffile

from src.utils.utils import scale_image_voxel_size


def import_labels_csv(path, csv_columns=('cell_ids', 'cell_labels')):
    cell_ids, cell_labels = [], []
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=csv_columns)
        reader = list(reader)
        for row in reader[1:]:
            cell_ids.append(row[csv_columns[0]])
            cell_labels.append(row[csv_columns[1]])

    return np.array(cell_ids, dtype='int32'), np.array(cell_labels, dtype='int32')


def read_tiff_voxel_size(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags

    # parse X, Y resolution
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    # return voxel size
    return [z, y, x]


def load_tiff(path):
    voxel_size = read_tiff_voxel_size(path)
    stack = tifffile.imread(path)
    return stack, voxel_size


def create_add_stack(path, key, stack, voxel_size=None, mode='a'):
    with h5py.File(path, mode) as f:
        if voxel_size is not None:
            f.attrs['element_size_um'] = voxel_size
        f.create_dataset(key, data=stack, chunks=True, compression='gzip')


def load_raw(raw_path, mean_voxel_size=None):
    raw, voxel_size = load_tiff(raw_path)
    if mean_voxel_size is not None:
        raw = scale_image_voxel_size(raw, voxel_size, mean_voxel_size, order=2)
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    return raw


def load_segmentation(seg_path, flip, mean_voxel_size=None):
    seg_path = Path(seg_path)
    assert seg_path.exists(), f'file {seg_path} does not exist'

    seg, voxel_size = load_tiff(seg_path)
    if flip:
        seg = seg[:, ::-1]

    if mean_voxel_size is not None:
        seg = scale_image_voxel_size(seg, voxel_size, mean_voxel_size, order=0)
    return seg
