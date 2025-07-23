import torch
import torch.nn as nn
import numpy as np

class PoseLoss(nn.Module):
    """
    Loss function for PoseNet.
    """
    def __init__(self, scale=100, pretrain_epoch=20) -> None:
        super().__init__()
        self.MIN_LOG_STD = np.log(1e-4)
        self.scale = scale
        self.pretrain_epoch = pretrain_epoch
        self.critirion = nn.MSELoss(reduction="mean")
        self.gaussiannll = nn.GaussianNLLLoss(reduction="mean")

    def forward(self, y_pred, pred_logstd, y, epoch):
        if epoch <= self.pretrain_epoch:
            return self.critirion(y_pred, y) * self.scale
        pred_logstd = torch.clamp(pred_logstd, min=self.MIN_LOG_STD)
        var = torch.exp(2 * pred_logstd)
        return self.gaussiannll(y_pred, y, var)
