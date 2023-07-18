import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from data import *
from torch_snippets import *
from utils import *
from model import *
from parse import parse_args


def show_mask(image_1, mask):
    image = image_1.copy()
    image[:,:,0][mask[:,:]==8] = 0 # yellow  wall
    image[:,:,0][mask[:,:]==1] = 147 # deep pink 承重柱
    image[:,:,1][mask[:,:]==1] = 20
    image[:,:,2][mask[:,:]==1] = 255
    image[:,:,1][mask[:,:]==2] = 125 # purple lift room 
    image[:,:,2][mask[:,:]==3] = 125 # blue other room
    image[:,:,:][mask[:,:]==4] = 105 # dark hospital school and hotel room 
    image[:,:,:2][mask[:,:]==5] = 80 # red open area
    image[:,:,1:][mask[:,:]==6] = 130 # dark blue toilet 
    image[:,:,0][mask[:,:]==7] = 127 # green office room 
    image[:,:,1][mask[:,:]==7] = 255
    image[:,:,2][mask[:,:]==7] = 0
    return image

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'mps' if torch.backends.mps.is_built() else 'cpu'
    # device = 'cpu'
    with_boundary = True if args.with_boundary == '1' else 0 
    val_ds = SegData('test', args.o_dataset, device, append= '.png',  resize= args.resize, with_doundary= with_boundary, get_path=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn)

    saveID = args.saveID
    base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    image_path = './image/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    ensureDir(image_path)

    if args.modeltype == 'UNet':
        model = UNet(pretrained=True, out_channels=args.n_class, backbone = args.backbone, with_boundary=with_boundary).to(device)

    model, _ = restore_checkpoint(model, base_path, device)
    t = 0 
    for bx, data in enumerate(val_dl):
        if with_boundary:
            ims, masks, boundaries, img_path = data
            _masks, _boundaries = model(ims)
            image = read(img_path[0], 1)
            image = cv2.resize(image, (args.resize, args.resize))
            _, _masks = torch.max(_masks, 1)
            _, _boundaries = torch.max(_boundaries, 1)

            # cv2.imwrite(f'{image_path}/{t}_image.png', image)
            # cv2.imwrite(f'{image_path}/{t}_Mask_G_type.png', masks[0].detach().cpu().numpy())
            # cv2.imwrite(f'{image_path}/{t}_Mask_G_boundary.png', boundaries[0].detach().cpu().numpy())
            # cv2.imwrite(f'{image_path}/{t}_Mask_P_type.png', _masks[0].detach().cpu().numpy())
            # cv2.imwrite(f'{image_path}/{t}_Mask_P_boundary.png', _boundaries[0].detach().cpu().numpy())
            img_p = show_mask(image, _masks[0].detach().cpu().numpy())
            cv2.imwrite(f'{image_path}/{t}_image_p.png', img_p)
            img_g = show_mask(image, masks[0].detach().cpu().numpy())
            cv2.imwrite(f'{image_path}/{t}_image_g.png', img_g)
        else:
            ims, masks, img_path = data
            _masks = model(ims)
            _, _masks = torch.max(_masks, 1)

            image = read(img_path[0], 1)
            image = cv2.resize(image, (args.resize, args.resize))

            img_p = show_mask(image, _masks[0].detach().cpu().numpy())
            cv2.imwrite(f'{image_path}/{t}_image_p.png', img_p)
            img_g = show_mask(image, masks[0].detach().cpu().numpy())
            cv2.imwrite(f'{image_path}/{t}_image_g.png', img_g)

            # cv2.imwrite(f'{image_path}/{t}_image.png', ims[0].detach().cpu().numpy())
            # cv2.imwrite(f'{image_path}/{t}_Mask_G.png', masks[0].detach().cpu().numpy())
            # cv2.imwrite(f'{image_path}/{t}_Mask_P.png', _masks[0].detach().cpu().numpy())
        t += 1 
        if t == 100:
            break


