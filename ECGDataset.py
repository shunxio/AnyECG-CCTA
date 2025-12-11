import torch
from torch.utils.data import Dataset
import numpy as np
import math

class ECGDataset(Dataset):
    def __init__(self, df, transform=None, strict_level=None,
                 use_augment=False, total_epochs=None, current_epoch=0):
        
        super().__init__()
        self.df = df
        self.file_paths = self.df.iloc[:, 11].values
        self.label_texts = self.df.iloc[:, [3, 4, 5, 6]]

        if strict_level is None:
            def label_processor(text):
                return 0 if text == "未见明显狭窄" else 1
        elif strict_level == "yes":
            def label_processor(text):
                return 0 if ("未见明显狭窄" in text or "轻度狭窄" in text or "中度狭窄" in text) else 1
        else:
            raise ValueError("strict_level 只允许为 None 或 'yes'")

        self.labels = self.label_texts.applymap(label_processor).values

        self.transform = transform
        self.use_augment = use_augment
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch

        print(f"[ECGDataset] 初始化完成，共 {len(self.df)} 条记录，增强：{'开启' if use_augment else '关闭'}")

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        ecg_file_path = self.file_paths[idx]

        try:
            signal = np.load(ecg_file_path)
            if signal.shape[1] == 12:
                signal = signal.T
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-8)
        except Exception as e:
            print(f"[警告] 文件加载失败：{ecg_file_path}, 错误：{e}")
            signal = np.zeros((12, 5000))
            label_tensor = torch.zeros(4, dtype=torch.float)

        signal_tensor = torch.tensor(signal, dtype=torch.float)

        if self.use_augment:
            if self.total_epochs is not None:
                cosine_scale = 0.5 * (1 + math.cos(math.pi * self.current_epoch / self.total_epochs))
                intensity = 0.5 + 0.5 * cosine_scale  # 从 1 → 0.5
            else:
                intensity = 1.0
            signal_aug = self.apply_augmentation(signal_tensor.clone(), intensity)
        else:
            signal_aug = signal_tensor.clone()

        if self.transform:
            signal_tensor = self.transform(signal_tensor)
            signal_aug = self.transform(signal_aug)

        return signal_tensor, signal_aug, label_tensor

    def apply_augmentation(self, signal, intensity=1.0):
        C, L = signal.shape

        if torch.rand(1) < 0.8 * intensity:
            shift = int(torch.randint(-int(0.1 * L), int(0.1 * L) + 1, (1,)))
            signal = torch.roll(signal, shifts=shift, dims=-1)

        if torch.rand(1) < 0.8 * intensity:
            scale = torch.empty(1).uniform_(0.9, 1.1)
            signal = signal * scale

        if torch.rand(1) < 0.5 * intensity:
            noise = torch.randn_like(signal) * 0.02 * intensity
            signal = signal + noise.clamp(-0.1, 0.1)

        if torch.rand(1) < 0.3 * intensity:
            mask_len = int(L * np.random.uniform(0.02, 0.1))
            start = np.random.randint(0, L - mask_len)
            signal[:, start:start + mask_len] = 0.0

        return signal
