import h5py
import napari

from src.utils.io import load_raw, load_segmentation


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
    labels_patches = seg_patches * labels[:, None, None, None]
    # start viewer
    viewer = napari.Viewer()
    viewer.add_image(raw_patches, contrast_limits=(0, 1))
    viewer.add_labels(labels_patches)
    return viewer
