import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='dataset1',
                        help='Choose a dataset')
    parser.add_argument('--modeltype', type=str, default='UNet',
                        help='which model wanna use.')
    parser.add_argument('--n_class', type=int, default=12,
                        help='no. class')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epoch.')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--max2keep', type=int, default=10,
                        help='max checkpoints to keep')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size.')
    return parser.parse_args()
