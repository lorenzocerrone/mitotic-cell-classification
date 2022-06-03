import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.classification.dataloader import PatchDataset2D, get_cv_splits
from src.classification.model import MitoticNet


def main():
    cv_splits = get_cv_splits('/home/lcerrone/data/Mitotic-cells/raw/')
    split = cv_splits[0]

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
    train_loader = DataLoader(train_dataset, batch_size=30, sampler=sampler, num_workers=20)

    val_loader = DataLoader(val_dataset, batch_size=30, num_workers=20)

    logger = pl_loggers.TensorBoardLogger('./logs')

    model = MitoticNet()

    trainer = pl.Trainer(accelerator='gpu', logger=logger, devices=1)
    trainer.fit(model, train_loader, val_loader)
