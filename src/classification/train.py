import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import csv

from src.classification.dataloader import PatchDataset2D, get_cv_splits
from src.classification.model import MitoticNet, aggregate_results

default_config = {'datadir': '/home/lcerrone/data/Mitotic-cells/raw/',
                  'split': 0,
                  'batch_size': 30,
                  'logdir': './logs'}


def train(config=None):
    config = config if config is not None else default_config

    cv_splits = get_cv_splits(config['datadir'])
    split = cv_splits[config['split']]

    m_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomAutocontrast(),
                                       # transforms.RandomErasing(),
                                       # transforms.RandomPerspective(),
                                       transforms.RandomRotation((0, 360)),
                                       transforms.GaussianBlur(kernel_size=5, sigma=(0.001, 2.)),
                                       transforms.Normalize((0.11714157462120056,), (0.13272494077682495,))
                                       ])

    normalize = transforms.Normalize((0.11714157462120056,), (0.13272494077682495,))

    train_dataset = PatchDataset2D(split['train'], use_cache=True, transforms=m_transforms)
    val_dataset = PatchDataset2D(split['val'], use_cache=True, transforms=normalize)

    sampler = WeightedRandomSampler(train_dataset.compute_weights(), len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=20)

    val_loader = DataLoader(val_dataset, batch_size=30, num_workers=20)

    logger = pl_loggers.TensorBoardLogger(config['logdir'])

    model = MitoticNet()

    trainer = pl.Trainer(accelerator='gpu', logger=logger, devices=1)
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
    all_predictions = []
    for data in test_loader:
        _, _, (pred, y, meta) = model.generic_step(data)
        all_predictions.append([pred, y, meta])

    results = aggregate_results(all_predictions)

    return results


def simple_predict(stack_path, model_path):
    normalize = transforms.Normalize((0.11714157462120056,), (0.13272494077682495,))
    test_dataset = PatchDataset2D([stack_path], use_cache=True, transforms=normalize)
    test_loader = DataLoader(test_dataset, batch_size=30, num_workers=20)

    model = MitoticNet()
    model = model.load_from_checkpoint(model_path)
    results = compute_predictions(model, test_loader)

    for key, result in results.items():
        out_path = key.replace('.h5', '_predictions.csv')
        export_labels_csv(result['cell_idx'], result['predictions'], path=out_path)
    return results
