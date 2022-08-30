import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class Model(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.feature_size, 1)

    def forward(self, features, labels=None):
        x = features.float()
        x = self.dropout(x)
        logits = self.out_proj(x)

        prob = torch.sigmoid(logits)
        if labels is not None:

            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob
        else:
            return prob
        return x
