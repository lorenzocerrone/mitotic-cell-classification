import h5py
import napari
import numpy as np
from magicgui import magicgui

from src.utils.io import load_raw, load_segmentation
from src.utils.utils import map_cell_features2segmentation


def basic_visualizer(raw_path, segmentation_path, flip):
    raw = load_raw(raw_path=raw_path)
    segmentation = load_segmentation(seg_path=segmentation_path, flip=flip)
    viewer = napari.Viewer()
    viewer.add_image(raw)
    viewer.add_labels(segmentation)
    return viewer


def patches_visualizer(patches_path):
    with h5py.File(patches_path, 'r') as f:
        raw_patches = f['raw_patches'][...]
        seg_patches = f['seg_patches'][...]
        labels = f['labels'][...]

    labels += 1
    if seg_patches.ndim == 3:
        labels_patches = seg_patches * labels[:, None, None]
    elif seg_patches.ndim == 4:
        labels_patches = seg_patches * labels[:, None, None, None]
    else:
        raise ValueError

    # start viewer
    viewer = napari.Viewer()
    viewer.add_image(raw_patches, contrast_limits=(0, 1))
    viewer.add_labels(labels_patches)
    return viewer


def predictions_visualizer(stain_path, patches_path, predictions, confidence=None):
    with h5py.File(stain_path, 'r') as f:
        raw = f['raw'][...]
        seg = f['segmentation'][...]

    with h5py.File(patches_path, 'r') as f:
        cell_idx = f['cell_idx'][...]
        labels = f['labels'][...]

    labels_not_empty = np.sum(labels) > 0.1

    viewer = napari.Viewer()
    viewer.add_image(raw)

    predictions_image = map_cell_features2segmentation(seg, cell_idx, predictions, data_type='float64')
    viewer.add_image(predictions_image, visible=True)

    if confidence is not None:
        predictions_image = map_cell_features2segmentation(seg, cell_idx, confidence, data_type='float64')
        viewer.add_image(predictions_image, visible=False)

    if labels_not_empty:
        labels_image = map_cell_features2segmentation(seg, cell_idx, labels)
        viewer.add_labels(labels_image, visible=False)

        errors = np.not_equal(labels, predictions)
        errors = np.where(errors, 7, 6)
        errors_image = map_cell_features2segmentation(seg, cell_idx, errors)

        viewer.add_labels(errors_image)
