import os
import math
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob


class MaskDataset(Dataset):
    def __init__(self, path, ss, ts):
        paths = glob(os.path.join(path, '*'))
        self.img_paths = sorted([path for path in paths if '.jpg' in path])
        self.mask_paths = [os.path.join(path.split('.')[0], '.png') for path in self.img_paths]
        self.ss = ss
        self.ts = ts
        self.transform = transforms.Compose([
            transforms.Resize(ss),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        print(f'[trainset] {len(self.img_paths)} images.')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L').resize((self.ss, self.ss)))
        mask[mask > 0] = 1

        return img, mask


def get_source_loader(args):
    dataset = MaskDataset(args.trainset, args.ss, args.ts)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.worker_num)

    return loader
