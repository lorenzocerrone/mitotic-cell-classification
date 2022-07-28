import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import csv
import torch

from src.classification.dataloader import PatchDataset, get_cv_splits
from src.classification.model import MitoticNet, aggregate_results
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

default_config = {'datadir': '/home/lcerrone/data/Mitotic-cells/raw/',
                  'split': 0,
                  'batch_size': 30,
                  'sub_epoch': 8,
                  'model': {'model_family': 'adapt_resnet',
                            'model_name': 'resnet50',
                            'pretrained': True,
                            'w_bias': 1.0,
                            'lr': 1e-3,
                            'wd': 1e-4,
                            'use_scheduler': False,
                            },
                  'logdir': './logs'}

"""
'model': {'model_family': 'adapt_convnext',
        'model_name': 'tiny',
        'pretrained': True,
        'w_bias': 0.1,
        'lr': 1e-3,
        'wd': 1e-5,
        'use_scheduler': False,
        },
"""


def checkpoint_callbacks():
    common_checkpoints_pattern: str = '{epoch:03d}_{val_acc:.2f}_{val_cm_diag:.2f}'
    cp1 = ModelCheckpoint(filename=f"last_{common_checkpoints_pattern}")
    cp2 = ModelCheckpoint(monitor='val_acc',
                          filename=f"best_acc_{common_checkpoints_pattern}",
                          mode='max')
    cp3 = ModelCheckpoint(monitor='val_cm_diag',
                          filename=f"best_cm_diag_{common_checkpoints_pattern}",
                          save_top_k=5,
                          mode='max')
    return [cp1, cp2, cp3, LearningRateMonitor()]


def train_transforms():
    t = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            # transforms.RandomErasing(),
                            # transforms.RandomRotation((0, 360)),
                            # transforms.GaussianBlur(kernel_size=5, sigma=(0.001, 2.)),
                            # transforms.Normalize(dataset_mean, dataset_std),
                            ])
    return t


def val_transforms():
    # t = transforms.Compose([transforms.Normalize(dataset_mean, dataset_std),
    #                        ])
    return None


def train(config=None):
    config = config if config is not None else default_config

    cv_splits = get_cv_splits(config['datadir'])
    split = cv_splits[config['split']]

    t_transforms = train_transforms()
    v_transforms = None  # val_transforms()
    train_dataset = PatchDataset(split['train'], transforms=t_transforms, load_seg=True)
    val_dataset = PatchDataset(split['val'], transforms=v_transforms, load_seg=True)

    sampler = WeightedRandomSampler(train_dataset.compute_weights(),
                                    len(train_dataset)//config['sub_epoch'],
                                    replacement=True)
    # sampler = WeightedRandomSampler(train_dataset.compute_weights(), 300, replacement=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              sampler=sampler,
                              num_workers=10,
                              persistent_workers=True)

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=config['batch_size'],
    #                           num_workers=10,
    #                           persistent_workers=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=30,
                            num_workers=10,
                            persistent_workers=True)

    exp_name = f"split:{config['split']}_model:{config['model']['model_name']}:{config['model']['w_bias']}"
    logger = pl_loggers.TensorBoardLogger(config['logdir'], name=exp_name)

    model = MitoticNet(config['model'])

    callbacks = checkpoint_callbacks()

    trainer = pl.Trainer(accelerator='gpu',
                         logger=logger,
                         max_epochs=20 * config['sub_epoch'],
                         devices=1,
                         log_every_n_steps=25,
                         callbacks=callbacks
                         )
    trainer.fit(model, train_loader, val_loader)


def export_labels_csv(cell_ids, cell_labels, path, csv_columns=('Label', 'Parent Label')):
    label_data = []
    for c_ids, c_l in zip(cell_ids, cell_labels):
        label_data.append({csv_columns[0]: c_ids, csv_columns[1]: c_l})

    with open(path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for data in label_data:
            writer.writerow(data)


def compute_predictions(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_predictions = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            raw, y, meta = data
            raw = raw.to(device)
            y = y.to(device)
            _, _, (pred, out, y) = model.generic_step(raw, y)
            pred, out, y = pred.cpu(), out.cpu(), y.cpu()
            all_predictions.append([pred, out, y, meta])

        results = aggregate_results(all_predictions)

    return results


def simple_predict(stack_path, model_paths, config=None):
    config = config if config is not None else default_config

    v_transforms = None  # val_transforms()

    test_dataset = PatchDataset([stack_path], transforms=v_transforms, load_seg=True)
    test_loader = DataLoader(test_dataset, batch_size=30, num_workers=10)

    results = {'outputs': {}}
    for model_path in model_paths:
        model = MitoticNet(config['model'])
        model = model.load_from_checkpoint(model_path)
        _results = compute_predictions(model, test_loader)
        results['outputs'][str(model_path)] = _results[stack_path]['outputs']

    results['cell_idx'] = _results[stack_path]['cell_idx']
    results['labels'] = _results[stack_path]['labels']

    """
    for key, result in results.items():
        out_path = key.replace('.h5', '_predictions.csv')
        export_labels_csv(result['cell_idx'], result['predictions'], path=out_path)
    """
    return results
