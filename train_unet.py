import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from data import *
from torch_snippets import *
from utils import *
from model import *
from parse import parse_args

if __name__ == '__main__':
    args = parse_args()
    trn_ds = SegData('train', args.dataset)
    val_ds = SegData('test', args.dataset)
    trn_dl = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, collate_fn=val_ds.collate_fn)

    saveID = args.saveID
    checkpoint_buffer = []
    base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    ensureDir(base_path)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_built() else 'cpu'
    if args.modeltype == 'UNet':
        model = UNet(pretrained=True, out_channels=args.n_class).to(device)
    criterion = UnetLoss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_epochs = args.epoch

    log = Report(n_epochs)

    best = 0
    for ex in range(n_epochs):
        N = len(trn_dl)
        running_loss, num_batches, running_acc = 0, 0, 0
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')
            running_loss += loss
            running_acc += acc

            num_batches += 1

        perf_str = 'Epoch %d: train==[%.5f %.5f]' % (
            ex, running_loss / num_batches,
            running_acc / num_batches)
        with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(perf_str + "\n")

        # evaluate the model
        if (ex + 1) % args.verbose == 0:
            N = len(val_dl)
            running_loss, num_batches, running_acc = 0, 0, 0
            for bx, data in enumerate(val_dl):
                loss, acc = validate_batch(model, data, criterion)
                running_loss += loss
                running_acc += acc

                log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')
                if acc > best:
                    checkpoint_buffer = save_checkpoint(model, ex, base_path, checkpoint_buffer, args.max2keep)
                    best = acc
                num_batches += 1

            perf_str = 'Valid set Epoch %d: train==[%.5f %.5f]' % (
                ex, running_loss / num_batches,
                running_acc / num_batches)
            with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
                f.write(perf_str + "\n")

    print("train_acc:", np.mean([v for pos, v in log.trn_acc]), "|val_acc:", np.mean([v for pos, v in log.val_acc]))

    log.plot_epochs(['trn_loss', 'val_loss'])

    ims, masks = next(iter(val_dl))
    output = model(ims)
    _, _masks = torch.max(output, 1)
    plt.imshow(ims[0].permute(1, 2, 0).detach().cpu()[:, :, 0], title="Original Image")
    plt.imshow(masks.permute(1, 2, 0).detach().cpu()[:, :, 0], title="Original Mask")
    plt.imshow(_masks.permute(1, 2, 0).detach().cpu()[:, :, 0], title="Predicated Mask")
