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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'mps' if torch.backends.mps.is_built() else 'cpu'
    # device = 'cpu'
    with_boundary = True if args.with_boundary == '1' else 0 
    trn_ds = SegData('train', args.dataset, device, append= '.png', resize= args.resize, with_doundary= with_boundary)
    val_ds = SegData('test', args.dataset, device, append= '.png',  resize= args.resize, with_doundary= with_boundary)
    trn_dl = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, collate_fn=val_ds.collate_fn)

    saveID = args.saveID
    checkpoint_buffer = []
    base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    image_path = './image/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    ensureDir(base_path)
    ensureDir(image_path)

    if args.modeltype == 'UNet':
        model = UNet(pretrained=True, out_channels=args.n_class, backbone = args.backbone, with_boundary=with_boundary).to(device)
    criterion = UnetLoss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_epochs = args.epoch

    log = Report(n_epochs)

    best = 0
    best_epoch = 0 
    for ex in range(n_epochs):
        N = len(trn_dl)
        running_loss, num_batches, running_acc = 0, 0, 0
        running_loss_b, running_loss_type, running_acc_type, running_acc_b = 0, 0, 0, 0
        for bx, data in enumerate(trn_dl):
            if not with_boundary:
                loss, acc = train_batch(model, data, optimizer, criterion)
                log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')
            else:
                loss_type, loss_b, acc_type, acc_b = train_batch_boundary(model, data, optimizer, criterion)
                log.record(ex + (bx + 1) / N, trn_loss_type=loss_type, trn_acc_type =acc_type, trn_loss_b = loss_b, trn_acc_b = acc_b, end='\r')
                running_loss_b += loss_b
                running_loss_type += loss_type
                running_acc_type += acc_type
                running_acc_b += acc_b

            running_loss += loss if not with_boundary else loss_type + 1.5*loss_b
            running_acc += acc if not with_boundary else (acc_type + acc_b)/2

            num_batches += 1

        if not with_boundary:
            perf_str = 'Epoch %d: train loss %.5f  train acc %.5f]' % (
                ex, running_loss / num_batches,
                running_acc / num_batches)
            with open(base_path + '/stats_{}.txt'.format(args.saveID), 'a') as f:
                f.write(perf_str + "\n")
        else:
            perf_str = 'Epoch %d: train loss %0.5f ==[%.5f %.5f] train acc %0.5f ==[%.5f %.5f] ' % (
                ex, running_loss / num_batches, running_loss_type/num_batches, running_loss_b/num_batches,
                running_acc / num_batches, running_acc_type / num_batches, running_acc_b / num_batches)
            with open(base_path + '/stats_{}.txt'.format(args.saveID), 'a') as f:
                f.write(perf_str + "\n")

        # evaluate the model
        if (ex + 1) % args.verbose == 0:
            N = len(val_dl)
            running_loss, num_batches, running_acc = 0, 0, 0
            running_loss_b, running_loss_type, running_acc_type, running_acc_b = 0, 0, 0, 0
            for bx, data in enumerate(val_dl):
                if with_boundary:
                    loss_type, loss_b, acc_type, acc_b = validate_batch_boundary(model, data, criterion)
                    log.record(ex + (bx + 1) / N, val_loss_type=loss_type, val_acc_type =acc_type, val_loss_b = loss_b, val_acc_b = acc_b, end='\r')
                    running_loss_b += loss_b
                    running_loss_type += loss_type
                    running_acc_type += acc_type
                    running_acc_b += acc_b
                else:
                    loss, acc = validate_batch(model, data, criterion)
                    log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')

                running_loss += loss if not with_boundary else loss_type + 1.5*loss_b
                running_acc += acc if not with_boundary else (acc_type + acc_b)/2

                num_batches += 1
            if with_boundary and (running_acc_type / num_batches) > best:
                checkpoint_buffer = save_checkpoint(model, ex, base_path, checkpoint_buffer, args.max2keep)
                best = (running_acc_type / num_batches)
                best_epoch = ex 
                
            if not with_boundary and (running_acc / num_batches) > best:
                checkpoint_buffer = save_checkpoint(model, ex, base_path, checkpoint_buffer, args.max2keep)
                best = (running_acc / num_batches)
                best_epoch = ex 

                
            if not with_boundary:
                perf_str = 'Epoch %d: valid loss %.5f valid acc %.5f]' % (
                    ex, running_loss / num_batches,
                    running_acc / num_batches)
                with open(base_path + '/stats_{}.txt'.format(args.saveID), 'a') as f:
                    f.write(perf_str + "\n")
            else:
                perf_str = 'Epoch %d: valid loss %0.5f ==[%.5f %.5f] valid acc %0.5f ==[%.5f %.5f] ' % (
                    ex, running_loss / num_batches, running_loss_type/num_batches, running_loss_b/num_batches,
                    running_acc / num_batches, running_acc_type / num_batches, running_acc_b / num_batches)
                with open(base_path + '/stats_{}.txt'.format(args.saveID), 'a') as f:
                    f.write(perf_str + "\n")

    print("train_acc:", np.mean([v for pos, v in log.trn_acc]), "|val_acc:", np.mean([v for pos, v in log.val_acc]))
    if not with_boundary:
        log.plot_epochs(['trn_loss', 'val_loss'], path=image_path+'/train_plot.png')
    else:
        log.plot_epochs(['trn_loss_type', 'trn_acc_type','val_loss_type', 'val_acc_type'], path=image_path+'/train_plot_type.png')
        log.plot_epochs(['trn_loss_b', 'trn_acc_b', 'val_loss_b', 'val_acc_b'], path=image_path+'/train_plot_boundary.png')
    # log.plot_epochs(['trn_loss', 'val_loss'])
    # model = restore_best_checkpoint(best_epoch, model, base_path, device)
    # ims, masks = next(iter(val_dl))
    # output = model(ims)
    # _, _masks = torch.max(output, 1)
    # plt.imsave(image_path+'/Original_Image.png', ims[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
    # plt.imsave(image_path+'/Original_Mask.png', masks.permute(1, 2, 0).detach().cpu()[:, :, 0])
    # plt.imsave(image_path+'/Predicated_Mask.png', _masks.permute(1, 2, 0).detach().cpu()[:, :, 0])
