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
    threshold = np.max(cm) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure, cm


class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=7, stride=3, bias=False)
        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(800, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x


def adapt_resnet(pretrained=True, model_name='resnet18'):
    models_dict = {'resnet18': [torchvision.models.resnet18, 64, 512],
                   'resnet34': [torchvision.models.resnet34, 64, 512],
                   'resnet50': [torchvision.models.resnet50, 64, 2048],
                   }

    _model, conv_out, fn_in = models_dict[model_name]
    conv1 = nn.Conv2d(4, conv_out, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained:
        model = _model(pretrained=True)

        with torch.no_grad():
            conv1.weight[:, :3] = model.conv1.weight.clone()

        model.fc = nn.Linear(fn_in, 1)

    else:
        model = _model(num_classes=1)

    model.conv1 = conv1
    return model


def adapt_convnext(pretrained=True, model_name='tiny'):
    models_dict = {'tiny': [torchvision.models.convnext_tiny, 96, 768],
                   'small': [torchvision.models.convnext_small, 96, 768],
                   'base': [torchvision.models.convnext_base, 128, 1024],
                   'large': [torchvision.models.convnext_large, 192, 1536],
                   }

    _model, conv_out, fn_in = models_dict[model_name]

    conv1 = torchvision.ops.misc.ConvNormActivation(4, conv_out,
                                                    kernel_size=4,
                                                    stride=4,
                                                    padding=0,
                                                    norm_layer=partial(LayerNorm2d, eps=1e-6),
                                                    activation_layer=None,
                                                    bias=True, )

    if pretrained:
        model = _model(pretrained=True)
        model.classifier[2] = nn.Linear(fn_in, 1)

        with torch.no_grad():
            conv1[0].weight[:, :3] = model.features[0][0].weight.clone()

    else:
        model = _model(num_classes=1)

    model.features[0] = conv1
    return model


class MitoticNet(pl.LightningModule):
    validation_predictions: dict

    def __init__(self, config):
        super(MitoticNet, self).__init__()

        if config['model_family'] == 'adapt_resnet':
            _model = adapt_resnet
        else:
            _model = adapt_convnext

        self.model = _model(pretrained=config['pretrained'],
                            model_name=config['model_name'])
        # self.model = BasicNet()

        self.w_bias = config['w_bias']

        self._lr = config['lr']
        self._wd = config['wd']
        self.use_scheduler = config['use_scheduler']

        self.accuracy = torchmetrics.Accuracy(num_classes=2)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2)
        self.save_hyperparameters()

    def configure_optimizers(self):
        lr = self._lr
        wd = self._wd
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=lr,
                                                            steps_per_epoch=1179,
                                                            epochs=10)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.model(x)

    def generic_step(self, raw, y, *args):
        # to generalize
        logits = self.model(raw)
        logits = torch.squeeze(logits)
        prob = torch.sigmoid(logits)

        w = torch.where(y > 0.5, self.w_bias, 1.0)
        loss = F.binary_cross_entropy(prob, y.float(), weight=w)
        # loss = F.binary_cross_entropy(prob, y.float())

        pred = prob > 0.5
        acc = self.accuracy(pred, y)
        return loss, acc, (pred, prob, y)

    def training_step(self, batch, batch_idx):
        loss, acc, _ = self.generic_step(*batch)

        self.log('train_loss', loss, batch_size=batch[0].shape[0])
        self.log('train_acc', acc, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, (pred, prob, y) = self.generic_step(*batch)

        self.log('val_loss', loss, batch_size=pred.shape[0])

        return pred, prob, y, batch[2]

    def on_validation_epoch_start(self) -> None:
        self.validation_predictions = {}

    def validation_epoch_end(self, outputs) -> None:
        results = aggregate_results(outputs)
        acc_mean, cm_diag_mean = 0, 0
        batch_size = 0

        for path, res in results.items():
            name = path.split('/')[-1]
            pred = torch.Tensor(res['predictions']).long()
            batch_size = pred.shape[0]

            lab = torch.Tensor(res['labels']).long()
            self.confusion_matrix = self.confusion_matrix.cpu()
            cm = self.confusion_matrix(pred.cpu(), lab.cpu()).cpu().numpy()
            figure, cm = plot_confusion_matrix(cm)
            cm_diag = (cm[0, 0] + cm[1, 1]) / 2
            cm_diag_mean += cm_diag

            acc = self.accuracy(pred, lab)
            acc_mean += acc

            self.logger.experiment.add_figure(f'conf matrix: {name}',
                                              figure,
                                              global_step=self.current_epoch)
            self.log(f'val: {name}', acc, batch_size=1)
            self.log(f'val cm diagonal: {name}', cm_diag, batch_size=1)

        self.log('val_acc', acc_mean / len(results.keys()), batch_size=batch_size)
        self.log('val_cm_diag', cm_diag_mean / len(results.keys()), batch_size=batch_size)


def aggregate_results(outputs):
    results = {}
    for pred, out, y, meta in outputs:
        for path, cell_idx, _pred, _out, _y in zip(meta['path'], meta['cell_idx'], pred, out, y):
            if path not in results:
                results[path] = {'cell_idx': [], 'predictions': [], 'outputs': [], 'labels': []}

            results[path]['cell_idx'].append(cell_idx.item())
            results[path]['predictions'].append(_pred.item())
            results[path]['outputs'].append(_out.item())
            results[path]['labels'].append(_y.item())
    return results
