from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def get_cv_splits(source_root: str,
                  number_splits: int = 5,
                  seed: int = 0) -> dict:
    base_path = Path(source_root)
    list_stacks = list(base_path.glob('**/*.h5'))
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


class PatchDataset2D(Dataset):
    def __init__(self, list_paths, use_cache=False, transforms=None, h5_cache=True):
        global_list_stack, global_num_nuclei, labels = self.get_samples_mapping(list_paths)

        self.labels = labels
        self.z_mid_point = self.get_z_mid_point(list_paths[0])
        self.global_num_nuclei = global_num_nuclei
        self.global_list_stack = global_list_stack

        self.use_cache = use_cache
        self.cache = {}

        self.transforms = transforms

        self.h5_cache = h5_cache
        self.data_cache = {}

    @staticmethod
    def get_z_mid_point(path):
        with h5py.File(path, 'r') as f:
            x = f['raw_patches'].shape[1] // 2
        return x

    def load_from_file(self, idx):
        cell_pos, path, cell_idx = self.global_list_stack[idx]
        with h5py.File(path, 'r') as f:
            raw = f['raw_patches'][cell_pos, self.z_mid_point, ...]

        raw = torch.from_numpy(raw).float()
        return raw, {'cell_pos': cell_pos, 'path': str(path), 'cell_idx': cell_idx}

    def load_from_data_cache(self, idx):
        cell_pos, path, cell_idx = self.global_list_stack[idx]
        if path not in self.data_cache:
            with h5py.File(path, 'r') as f:
                raw = f['raw_patches'][:, self.z_mid_point, ...]
                raw = torch.from_numpy(raw).float()
                self.data_cache[path] = raw

        raw = self.data_cache[path][cell_pos].clone()
        return raw, {'cell_pos': cell_pos, 'path': str(path), 'cell_idx': cell_idx}

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
        return global_list_stack, len(global_list_stack), torch.cat(all_labels)

    def get(self, idx):

        if self.h5_cache:
            raw, meta = self.load_from_data_cache(idx)
        else:
            raw, meta = self.load_from_file(idx)

        raw = torch.unsqueeze(raw, 0)
        # seg = torch.from_numpy(seg.astype('int32')).float()

        # return torch.stack([raw, seg], 0), label
        return raw, self.labels[idx], meta

    def compute_weights(self, class_weights=None):
        if class_weights is None:
            _, class_weights = torch.unique(self.labels, return_counts=True)
            class_weights = 1 / class_weights

        weights = torch.where(self.labels == 0, class_weights[0], class_weights[1])
        return weights

    def get_from_cache(self, idx):
        if idx in self.cache:
            data = self.cache[idx]
        else:
            data = self.get(idx)
            self.cache[idx] = data

        return data[0].clone(), data[1], data[2]

    def __getitem__(self, idx):
        data, label, meta = self.get_from_cache(idx) if self.use_cache else self.get(idx)

        if self.transforms is not None:
            data = self.transforms(data)

        return data, label, meta

    def __len__(self):
        return self.global_num_nuclei
