from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.convnext import LayerNorm2d


def adapt_convnext():
    model = torchvision.models.convnext_small()

    model.features[0] = torchvision.ops.misc.ConvNormActivation(1,
                                                                96,
                                                                kernel_size=4,
                                                                stride=4,
                                                                padding=0,
                                                                norm_layer=partial(LayerNorm2d, eps=1e-6),
                                                                activation_layer=None,
                                                                bias=True, )
    model.classifier[2] = nn.Linear(768, 2)
    return model


class MitoticNet(pl.LightningModule):
    validation_predictions: dict

    def __init__(self):
        super(MitoticNet, self).__init__()
        self.model = adapt_convnext()

    def configure_optimizers(self):
        lr = 1e-3
        wd = 1e-6

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def _generic_step(self, batch):
        raw, y, meta = batch
        # to generalize
        out = self.model(raw)
        logits = torch.log_softmax(out, 1)
        loss = F.nll_loss(logits, y)

        pred = logits.max(1)[1]
        acc = torch.eq(pred, y).float().mean()
        return loss, acc, (pred, y, meta)

    def training_step(self, batch, batch_idx):
        loss, acc, _ = self._generic_step(batch)

        self.log('train_loss', loss, batch_size=batch[0].shape[0])
        self.log('train_acc', acc, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, (pred, y, meta) = self._generic_step(batch)

        self.log('val_loss', loss, batch_size=pred.shape[0])

        return pred, y, meta

    def on_validation_epoch_start(self) -> None:
        self.validation_predictions = {}

    def validation_epoch_end(self, outputs) -> None:
        results = {}
        for pred, y, meta in outputs:
            for path, cell_idx, _pred, _y in zip(meta['path'], meta['cell_idx'], pred, y):
                if path not in results:
                    results[path] = {'cell_idx': [], 'predictions': [], 'labels': []}

                results[path]['cell_idx'].append(cell_idx.item())
                results[path]['predictions'].append(_pred.item())
                results[path]['labels'].append(_y.item())

        for path, res in results.items():
            name = path.split('/')[-1]
            acc = np.mean(np.array(res['predictions']) == np.array(res['labels']))
            self.log(f'val: {name}', acc, batch_size=1)
