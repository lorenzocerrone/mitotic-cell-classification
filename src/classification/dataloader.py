from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


def get_cv_splits(source_root: str,
                  number_splits: int = 5,
                  seed: int = 0) -> dict:
    base_path = Path(source_root)
    list_stacks = list(base_path.glob('**/*patches.h5'))
    np.random.seed(seed)
    np.random.shuffle(list_stacks)

    splits = {i: {'val': [], 'train': []} for i in range(number_splits)}

    for i in range(number_splits):
        stage_split = np.array_split(list_stacks, number_splits)

        test_split = stage_split.pop(i).tolist()

        train_split = []
        for _split in stage_split:
            train_split += _split.tolist()

        splits[i]['train'] = train_split
        splits[i]['val'] = test_split
    return splits


class PatchDataset(Dataset):
    def __init__(self, list_paths, apply_norm=True, transforms=None, h5_cache=True, load_seg=False, n_slices=3):
        global_list_stack, global_num_nuclei, labels = self.get_samples_mapping(list_paths)

        self.labels = labels
        self.global_num_nuclei = global_num_nuclei
        self.global_list_stack = global_list_stack

        self.apply_norm = apply_norm
        self.cache = {}

        self.transforms = transforms

        self.load_seg = load_seg
        self.h5_cache = h5_cache
        self.data_cache = {}
        self.stats_cache = {}
        self.n_slices = n_slices

    def load_from_file(self, idx):
        cell_pos, path, cell_idx = self.global_list_stack[idx]
        with h5py.File(path, 'r') as f:
            # load raw
            raw = f['raw_patches'][cell_pos, ...]
            raw = torch.from_numpy(raw).float()

            # load seg
            if self.load_seg:
                seg_shape = f['seg_patches'].shape
                seg = f['seg_patches'][cell_pos, seg_shape[1]//2, ...]
                seg = seg[None, ...]
                seg = torch.from_numpy(seg.astype('int32')).float()
                # cat on axis 0 because cell is already gone
                raw = torch.cat([raw, seg], 0)

            mean, std = f.attrs['raw_patches_mean'], f.attrs['raw_patches_std']

        return raw, {'cell_pos': cell_pos,
                     'path': str(path),
                     'cell_idx': cell_idx,
                     'mean': mean,
                     'std': std}

    def load_from_data_cache(self, idx):
        cell_pos, path, cell_idx = self.global_list_stack[idx]
        if path not in self.data_cache:
            with h5py.File(path, 'r') as f:
                raw = f['raw_patches'][:, ...]
                raw = torch.from_numpy(raw).float()

                # load seg
                if self.load_seg:
                    seg_shape = f['seg_patches'].shape
                    seg = f['seg_patches'][:, seg_shape[1]//2, ...]
                    seg = seg[:, None, ...]
                    seg = torch.from_numpy(seg.astype('int32')).float()
                    # cat on axis 1 because 0 is the patch level
                    raw = torch.cat([raw, seg], 1)

                mean, std = f.attrs['raw_patches_mean'], f.attrs['raw_patches_std']

                self.data_cache[path] = raw
                self.stats_cache[path] = (mean, std)

        raw = self.data_cache[path][cell_pos].clone()
        mean, std = self.stats_cache[path]
        return raw, {'cell_pos': cell_pos,
                     'path': str(path),
                     'cell_idx': cell_idx,
                     'mean': mean,
                     'std': std}

    @staticmethod
    def get_samples_mapping(list_paths):
        global_list_stack = []
        all_labels = []

        for path in list_paths:
            with h5py.File(path, 'r') as f:
                cell_idx = f['cell_idx'][...]
                all_labels.append(torch.from_numpy(f['labels'][...]))
                list_stack = [(i, path, _cell_idx) for i, _cell_idx in enumerate(cell_idx)]

            global_list_stack += list_stack
        return global_list_stack, len(global_list_stack), torch.cat(all_labels).long()

    def get(self, idx):

        if self.h5_cache:
            raw, meta = self.load_from_data_cache(idx)
        else:
            raw, meta = self.load_from_file(idx)
        return raw, self.labels[idx], meta

    def compute_weights(self, class_weights=None):
        if class_weights is None:
            _, class_weights = torch.unique(self.labels, return_counts=True)
            #   class_weights = 1 / class_weights
            w0 = class_weights[1] / (class_weights[0] + class_weights[1])
            w1 = class_weights[0] / (class_weights[0] + class_weights[1])
        else:
            w0, w1 = class_weights

        weights = torch.where(self.labels == 0, w0, w1)
        return weights

    def __getitem__(self, idx):

        data, label, meta = self.get(idx)
        if self.transforms is not None:
            data = self.transforms(data)

        if self.apply_norm:
            data = F.normalize(data,
                               [meta['mean'], meta['mean'], meta['mean'], 0.],
                               [meta['std'], meta['std'], meta['std'], 1.])

        return data, label, meta

    def __len__(self):
        return self.global_num_nuclei

