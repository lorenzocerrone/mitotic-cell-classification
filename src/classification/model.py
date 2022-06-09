from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torchmetrics
from torchvision.models.convnext import LayerNorm2d
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names=('Normal', 'Mitotic')):
    """
    taken from
    https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    # Normalize the confusion matrix.
    norm = cm.sum(axis=1)[:, np.newaxis]
    num_mitotic = norm[1][0]
    tot_cells = np.sum(norm)
    cm = np.around(cm.astype('float') / (norm + 1e-16), decimals=3)

    figure = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: Total cells: {tot_cells}, Mitotic: {num_mitotic}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


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
        self.accuracy = torchmetrics.Accuracy(num_classes=2)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2)

    def configure_optimizers(self):
        lr = 1e-3
        wd = 1e-6

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def generic_step(self, raw, y):
        # to generalize
        out = self.model(raw)
        logits = torch.log_softmax(out, 1)
        loss = F.nll_loss(logits, y)

        pred = logits.max(1)[1]
        acc = self.accuracy(pred, y)
        return loss, acc, (pred, y)

    def training_step(self, batch, batch_idx):
        loss, acc, _ = self.generic_step(*batch)

        self.log('train_loss', loss, batch_size=batch[0].shape[0])
        self.log('train_acc', acc, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, (pred, y) = self.generic_step(*batch)

        self.log('val_loss', loss, batch_size=pred.shape[0])

        return pred, y, batch[2]

    def on_validation_epoch_start(self) -> None:
        self.validation_predictions = {}

    def validation_epoch_end(self, outputs) -> None:
        results = aggregate_results(outputs)

        for path, res in results.items():
            name = path.split('/')[-1]
            pred = torch.Tensor(res['predictions']).long()
            lab = torch.Tensor(res['labels']).long()
            self.confusion_matrix = self.confusion_matrix.cpu()
            cm = self.confusion_matrix(pred.cpu(), lab.cpu()).cpu().numpy()
            acc = self.accuracy(pred, lab)

            self.logger.experiment.add_figure(f'conf matrix: {name}',
                                              plot_confusion_matrix(cm),
                                              global_step=self.current_epoch)
            self.log(f'val: {name}', acc, batch_size=1)


def aggregate_results(outputs):
    results = {}
    for pred, y, meta in outputs:
        for path, cell_idx, _pred, _y in zip(meta['path'], meta['cell_idx'], pred, y):
            if path not in results:
                results[path] = {'cell_idx': [], 'predictions': [], 'labels': []}

            results[path]['cell_idx'].append(cell_idx.item())
            results[path]['predictions'].append(_pred.item())
            results[path]['labels'].append(_y.item())
    return results
