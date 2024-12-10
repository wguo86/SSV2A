import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_fs):
        self.img_fs = img_fs

    def __len__(self):
        return len(self.img_fs)

    def __getitem__(self, idx):
        img = Image.open(self.img_fs[idx])
        return np.array(img), self.img_fs[idx]

