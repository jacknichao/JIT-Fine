import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.args = args

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config, args)
        self.args = args

    def forward(self, inputs_ids, attn_masks,
                labels=None):
        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_masks)[0]
        logits = self.classifier(outputs)

        prob = torch.sigmoid(logits)
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob
        else:
            return prob

