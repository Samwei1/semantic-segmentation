from torch_snippets import *
import torch
from PIL import Image
import numpy as np
tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])


class SegData(Dataset):
    def __init__(self, split, dataset, device, append = '.png', resize = 256, with_doundary = False, get_path = False):
        self.items = stems(f'{dataset}/images_prepped_{split}')
        self.split = split
        self.dataset = dataset
        self.device = device
        self.append = append
        self.resize = resize 
        self.with_boundary = with_doundary
        self.get_path = get_path

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'{self.dataset}/images_prepped_{self.split}/{self.items[ix]}'+self.append, 1)
        image = cv2.resize(image, (self.resize, self.resize))
        mask = Image.open(f'{self.dataset}/annotations_prepped_{self.split}/{self.items[ix]}.png')
        mask = np.asarray(mask, dtype=np.uint8)
        mask = cv2.resize(mask, (self.resize, self.resize))
        img_path = f'{self.dataset}/images_prepped_{self.split}/{self.items[ix]}'+self.append
        if not self.with_boundary:
            if not self.get_path:
                return image, mask  
            else:
                return image, mask, img_path
        else:
            boundary = cv2.imread(f'{self.dataset}/annotations_prepped_boundary_{self.split}/{self.items[ix]}.png', 0)
            boundary = cv2.resize(boundary, (self.resize, self.resize))
            if not self.get_path:
                return image, mask, boundary  
            else:
                return image, mask, boundary, img_path

    def choose(self): return self[randint(len(self))]

    def collate_fn(self, batch):
        if not self.with_boundary:
            if self.get_path:
                ims, masks, img_path = list(zip(*batch)) 
            else:
                ims, masks = list(zip(*batch)) 
            
        else:
            if self.get_path:
                ims, masks, boundaries, img_path = list(zip(*batch))  
            else:
                ims, masks, boundaries = list(zip(*batch))
        ims = torch.cat([tfms(im.copy() / 255.)[None] for im in ims]).float().to(self.device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(self.device)
        if not self.with_boundary:
            if not self.get_path:
                return ims, ce_masks 
            else:
                return ims, ce_masks, img_path
        else:
            ce_boundary = torch.cat([torch.Tensor(boundary[None]) for boundary in boundaries]).long().to(self.device)
            if not self.get_path:
                return ims, ce_masks, ce_boundary  
            else:
                return ims, ce_masks, ce_boundary, img_path

