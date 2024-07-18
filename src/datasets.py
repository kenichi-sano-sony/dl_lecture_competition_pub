import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt

def preprocess_eeg(eeg_data, original_sfreq=1000, new_sfreq=250, baseline_interval=None):
    num_samples = int(eeg_data.shape[-1] * new_sfreq / original_sfreq)
    eeg_data = resample(eeg_data, num_samples, axis=-1)

    low_cutoff = 1
    high_cutoff = 40
    b, a = butter(4, [low_cutoff / (0.5 * new_sfreq), high_cutoff / (0.5 * new_sfreq)], btype='band')
    eeg_data = filtfilt(b, a, eeg_data, axis=-1)

    eeg_data = (eeg_data - np.mean(eeg_data, axis=-1, keepdims=True)) / np.std(eeg_data, axis=-1, keepdims=True)

    if baseline_interval is not None:
        baseline = np.mean(eeg_data[:, :, :baseline_interval], axis=-1, keepdims=True)
        eeg_data = eeg_data - baseline

    return eeg_data


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", original_sfreq=1000, new_sfreq=250, baseline_interval=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq
        self.baseline_interval = baseline_interval

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        eeg_data = self.X[i].numpy()
        eeg_data = preprocess_eeg(eeg_data, self.original_sfreq, self.new_sfreq, self.baseline_interval)
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
  
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]