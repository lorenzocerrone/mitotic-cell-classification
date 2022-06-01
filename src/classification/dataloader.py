from torch.utils.data import Dataset
import h5py
import torch


class PatchDataset2D(Dataset):
    def __init__(self, list_paths, use_cache=False):
        global_list_stack, global_num_nuclei = self.get_samples_mapping(list_paths)

        self.z_mid_point = self.get_z_mid_point(list_paths[0])
        self.global_num_nuclei = global_num_nuclei
        self.global_list_stack = global_list_stack
        self.use_cache = use_cache
        self.cache = {}

    @staticmethod
    def get_z_mid_point(path):
        with h5py.File(path, 'r') as f:
            x = f['raw_patches'].shape[1] // 2
        return x

    @staticmethod
    def get_samples_mapping(list_paths):
        global_list_stack = []

        for path in list_paths:
            with h5py.File(path, 'r') as f:
                num_nuclei = f['labels'].shape[0]
                list_stack = [(path, i) for i in range(num_nuclei)]

        global_list_stack += list_stack
        return global_list_stack, len(global_list_stack)

    def get(self, idx):
        path, cell_idx = self.global_list_stack[idx]
        with h5py.File(path, 'r') as f:
            raw = f['raw_patches'][cell_idx, self.z_mid_point, ...]
            seg = f['seg_patches'][cell_idx, self.z_mid_point, ...]
            label = f['labels'][cell_idx]

        raw = torch.from_numpy(raw).float()
        seg = torch.from_numpy(seg.astype('int32')).float()

        return torch.stack([raw, seg], 0), label

    def get_from_cache(self, idx):
        if idx in self.cache:
            data = self.cache[idx]
        else:
            data = self.get(idx)
            self.cache[idx] = data

        return data[0].clone(), data[1]

    def __getitem__(self, idx):
        data = self.get_from_cache(idx) if self.use_cache else self.get(idx)
        return data

    def __len__(self):
        return self.global_num_nuclei
