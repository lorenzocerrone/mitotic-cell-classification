from pathlib import Path

from math import ceil
import h5py
import numba
import numpy as np
import tqdm
from skimage.filters import gaussian

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


def get_slice(x, center, shape):
    return x[center[0] - shape[0] // 2:center[0] + ceil(shape[0] / 2),
             center[1] - shape[1] // 2:center[1] + ceil(shape[1] / 2),
             center[2] - shape[2] // 2:center[2] + ceil(shape[2] / 2)]


def build_patches2d(bboxes, raw, seg, label_mapping=None, shape=(1, 64, 64), sigma=1.0):
    list_raw, list_seg, list_gt, list_idx, list_bbox = [], [], [], [], []
    for i, (key, value) in enumerate(tqdm.tqdm(bboxes.items())):
        zmin, xmin, ymin, zmax, xmax, ymax = value
        center = [zmin + (zmax - zmin) // 2,
                  xmin + (xmax - xmin) // 2,
                  ymin + (ymax - ymin) // 2]

        raw_box = get_slice(raw, center, shape)
        seg_box = get_slice(seg, center, shape)

        if sigma is not None:
            raw_box = gaussian(raw_box, sigma)

        seg_mask = np.zeros_like(seg_box)
        seg_mask[seg_box == key] = 1
        seg_mask[seg_box != key] = 0

        list_raw.append(raw_box)
        list_seg.append(seg_mask)

        # append label
        if label_mapping is not None:

            if label_mapping[key] == 1:
                list_gt.append(1)
            elif label_mapping[key] == 10:
                list_gt.append(0)
            else:
                raise ValueError
        else:
            list_gt.append(0)

        # append idx and com
        list_idx.append(key)
        list_bbox.append(value)

    list_raw = np.array(list_raw)
    list_raw = (list_raw - np.min(list_raw)) / (np.max(list_raw) - np.min(list_raw))
    raw_mean, raw_std = np.mean(list_raw), np.std(list_raw)

    return {'raw_patches': list_raw,
            'seg_patches': np.array(list_seg),
            'labels': np.array(list_gt),
            'cell_bbox': np.array(list_bbox),
            'cell_idx': np.array(list_idx),
            }, (raw_mean, raw_std)


def pad_stack(stack, shape):
    return np.pad(stack, pad_width=((shape[0] // 2, shape[0] // 2),
                                    (shape[1] // 2, shape[1] // 2),
                                    (shape[2] // 2, shape[2] // 2)))


def process_tiff2h5(raw_path,
                    segmentation_path,
                    flip=False,
                    mean_voxel_size=(0.281, 0.126, 0.126)):
    raw_path = Path(raw_path)
    out_file = raw_path.parent / f'{raw_path.stem}.h5'

    # load files
    print('-processing raw stain...')
    raw = load_raw(raw_path, mean_voxel_size=mean_voxel_size)
    create_add_stack(path=out_file, key='raw', stack=raw, voxel_size=mean_voxel_size, mode='w')

    print('-processing segmentation...')
    seg = load_segmentation(segmentation_path, flip=flip, mean_voxel_size=mean_voxel_size)
    create_add_stack(path=out_file, key='segmentation', stack=seg, voxel_size=mean_voxel_size)
    return out_file, (raw, seg)


def process_data(h5path,
                 labels_csv_path=None,
                 shape=(0, 128, 128),
                 sigma=None,
                 slack=(2, 20, 20)):
    h5path = Path(h5path)

    with h5py.File(h5path, 'r') as f:
        raw = f['raw'][...]
        seg = f['segmentation'][...]

    # proces segmentation to get bbox
    seg_label = np.unique(seg)

    if labels_csv_path is None:
        label_mapping = None
    else:
        idx_labels, labels = import_labels_csv(labels_csv_path)
        label_mapping = create_features_mapping(idx_labels, labels, seg_label[1:])

    print('-add_patches')
    raw = pad_stack(raw, shape=shape)
    seg = pad_stack(seg, shape=shape)

    print('-building bboxes...')
    bboxes = get_bboxes(seg.astype('int64'), seg_label[1:].astype('int64'), slack=slack)

    print('-building patches...')
    patches, (raw_mean, raw_std) = build_patches2d(bboxes,
                                                   raw,
                                                   seg,
                                                   label_mapping,
                                                   shape=shape,
                                                   sigma=sigma)

    out_file = h5path.parent / f'{h5path.stem}_patches.h5'
    out_file.unlink(missing_ok=True)

    for key, value in patches.items():
        create_add_stack(path=out_file, key=key, stack=value)

    with h5py.File(out_file, 'a') as f:
        f.attrs['raw_patches_mean'] = raw_mean
        f.attrs['raw_patches_std'] = raw_std

    return out_file
