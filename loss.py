import torch
import torch.nn as nn
import torch.nn.functional as F

def mape_loss(pred, target):
    # pred, target: [B, 1], torch tenspr
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    return loss.mean()