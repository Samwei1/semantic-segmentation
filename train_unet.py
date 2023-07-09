import torch
import time
import numpy as np
from data import *
from torch_snippets import *
from utils import *
from model import *

if __name__ == '__main__':
    trn_ds = SegData('train')
    val_ds = SegData('test')
    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_built() else 'cpu'


    model = UNet().to(device)
    criterion = UnetLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 20

    log = Report(n_epochs)

    for ex in range(n_epochs):
        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')

        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')

    print("train_acc:", np.mean([v for pos, v in log.trn_acc]), "|val_acc:", np.mean([v for pos, v in log.val_acc]))

    log.plot_epochs(['trn_loss', 'val_loss'])

    ims, masks = next(iter(val_dl))
    output = model(ims)
    _, _masks = torch.max(output, 1)
    show(ims[0].permute(1, 2, 0).detach().cpu()[:, :, 0], title="Original Image")
    show(masks.permute(1, 2, 0).detach().cpu()[:, :, 0], title="Original Mask")
    show(_masks.permute(1, 2, 0).detach().cpu()[:, :, 0], title="Predicated Mask")






