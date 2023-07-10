from torch_snippets import *
import torch

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])


class SegData(Dataset):
    def __init__(self, split, dataset, device):
        self.items = stems(f'{dataset}/images_prepped_{split}')
        self.split = split
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'{self.dataset}/images_prepped_{self.split}/{self.items[ix]}.png', 1)
        image = cv2.resize(image, (224, 224))
        mask = read(f'{self.dataset}/annotations_prepped_{self.split}/{self.items[ix]}.png')
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    def choose(self): return self[randint(len(self))]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([tfms(im.copy() / 255.)[None] for im in ims]).float().to(self.device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(self.device)
        return ims, ce_masks
