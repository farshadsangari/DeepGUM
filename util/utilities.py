"""
Utilities of Project
"""

import argparse
import torch
import os
import torch.nn.utils as utils
from yaml import parse
import torch.nn as nn
import numpy as np


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

    all_dirs_path,

    momentum,


def get_args():
    """
    The argument that we have defined, will be used in training and evaluation(infrence) modes
    """
    parser = argparse.ArgumentParser(
        description="Arguemnt Parser of `Train` and `Evaluation` of RealNVP network"
    )

    parser.add_argument(
        "--lr", dest="lr", default=1e-4, type=float, help="Learning rate"
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=16,
        type=int,
        help="Batch size during training",
    )

    parser.add_argument(
        "--gamma", dest="gamma", default=0.08, type=float, help="Gamma value"
    )

    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        type=bool,
        help="Shuffle dataloaders or not",
    )

    parser.add_argument(
        "--max-epochs-vgg1",
        dest="max_epochs_vgg1",
        default=6,
        type=int,
        help="Max epochs of VGG1 model",
    )

    parser.add_argument(
        "--max-epochs-vgg2",
        dest="max_epochs_vgg2",
        default=2,
        type=int,
        help="Max epochs of VGG2 model",
    )

    parser.add_argument(
        "--max-epochs-em",
        dest="max_epochs_em",
        default=5,
        type=int,
        help="Max epochs of EM",
    )

    parser.add_argument(
        "--max-epochs-em-sgd",
        dest="max_epochs_em_sgd",
        default=5,
        type=int,
        help="Max epochs of EM SGD",
    )

    parser.add_argument(
        "--num-train-data",
        dest="num_train_data",
        default=1800,
        type=int,
        help="Number of Train images",
    )

    parser.add_argument(
        "--num-test-data",
        dest="num_test_data",
        default=120,
        type=int,
        help="Number of test images",
    )

    parser.add_argument(
        "--num-val-data",
        dest="num_val_data",
        default=80,
        type=int,
        help="Number of Validation images",
    )

    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        default=2,
        type=int,
        help="Number of Dataloader workers",
    )

    parser.add_argument("--cov", dest="cov", default=1, type=float, help="Covariance")

    parser.add_argument("--prior", dest="prior", default=0.95, type=float, help="prior")

    parser.add_argument(
        "--loss-threshold",
        dest="loss_threshold",
        default=1e-6,
        type=float,
        help="Threshold of the loss to stop",
    )

    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=1e-6,
        type=float,
        help="Weight decay",
    )

    parser.add_argument(
        "--momentum", dest="momentum", default=0.9, type=float, help="Momentum"
    )

    parser.add_argument(
        "--all-dirs-path",
        dest="all_dirs_path",
        default="/kaggle/working/CACD2000/*.jpg",
        type=str,
        help="directory of whole images",
    )

    options = parser.parse_args()

    return options


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save

    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group["params"], max_norm, norm_type)


def invert(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
