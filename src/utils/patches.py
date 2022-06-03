from pathlib import Path

import numba
import numpy as np
import tqdm

from src.utils.io import create_add_stack, import_labels_csv, load_raw, load_segmentation
from src.utils.utils import scale_image, create_features_mapping


@numba.njit(parallel=True)
def _get_bboxes(segmentation, labels_idx):
    shape = segmentation.shape

    bboxes = {}
    for idx in labels_idx:
        _x = np.array((shape[0], shape[1], shape[2], 0, 0, 0))
        bboxes[idx] = _x

    for z in numba.prange(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                idx = segmentation[z, x, y]
                if idx > 0:
                    zmin, xmin, ymin, zmax, xmax, ymax = bboxes[idx]

                    if z < zmin:
                        bboxes[idx][0] = z
                    if x < xmin:
                        bboxes[idx][1] = x
                    if y < ymin:
                        bboxes[idx][2] = y

                    if z > zmax:
                        bboxes[idx][3] = z
                    if x > xmax:
                        bboxes[idx][4] = x
                    if y > ymax:
                        bboxes[idx][5] = y
    return bboxes


def get_bboxes(segmentation, labels_idx, slack):
    bboxes = _get_bboxes(segmentation, labels_idx)
    slack = np.array([-slack[0], -slack[1], -slack[2],
                      slack[0], slack[1], slack[2]])
    bboxes_out = {}
    for key, values in bboxes.items():
        bboxes_out[int(key)] = values + slack
    return bboxes_out


def build_patches(bboxes, raw, seg, label_mapping, shape=(20, 64, 64)):
    list_raw, list_seg, list_gt, list_idx, list_bbox = [], [], [], [], []
    for i, (key, value) in enumerate(tqdm.tqdm(bboxes.items())):
        zmin, xmin, ymin, zmax, xmax, ymax = value
        raw_box = raw[zmin:zmax, xmin:xmax, ymin:ymax]
        seg_box = seg[zmin:zmax, xmin:xmax, ymin:ymax]
        if np.alltrue(raw_box.shape):
            # append raw
            raw_box = scale_image(raw_box, shape, raw_box.shape, order=2)
            seg_box = scale_image(seg_box, shape, seg_box.shape, order=0)

            seg_mask = np.zeros_like(seg_box)
            seg_mask[seg_box == key] = 1
            seg_mask[seg_box != key] = 0

            list_raw.append(raw_box)
            list_seg.append(seg_mask)

            # append label
            if label_mapping[key] == 1:
                list_gt.append(1)
            elif label_mapping[key] == 10:
                list_gt.append(0)
            else:
                raise ValueError

            # append idx and com
            list_idx.append(key)
            list_bbox.append(value)

    return {'raw_patches': np.array(list_raw),
            'seg_patches': np.array(list_seg),
            'labels': np.array(list_gt),
            'cell_bbox': np.array(list_bbox),
            'cell_idx': np.array(list_idx),
            }


def process_data(raw_path,
                 segmentation_path,
                 labels_csv_path,
                 flip=False,
                 shape=(20, 84, 84),
                 slack=(2, 20, 20),
                 mean_voxel_size=(0.281, 0.126, 0.126)):
    raw_path = Path(raw_path)
    out_file = raw_path.parent / f'{raw_path.stem}_patches.h5'

    # load files
    print('-processing raw stain...')
    raw = load_raw(raw_path, mean_voxel_size=mean_voxel_size)
    create_add_stack(path=out_file, key='raw', stack=raw, voxel_size=mean_voxel_size, mode='w')

    print('-processing segmentation...')
    seg = load_segmentation(segmentation_path, flip=flip, mean_voxel_size=mean_voxel_size)
    create_add_stack(path=out_file, key='segmentation', stack=seg, voxel_size=mean_voxel_size)

    idx_labels, labels = import_labels_csv(labels_csv_path)

    # proces segmentation to get bbox
    seg_label = np.unique(seg)

    label_mapping = create_features_mapping(idx_labels, labels, seg_label[1:])
    print('-building bboxes...')
    bboxes = get_bboxes(seg.astype('int64'), seg_label[1:].astype('int64'), slack=slack)
    print('-building patches...')
    patches = build_patches(bboxes, raw,
                            seg,
                            label_mapping,
                            shape=shape)

    for key, value in patches.items():
        create_add_stack(path=out_file, key=key, stack=value)

    return out_file
