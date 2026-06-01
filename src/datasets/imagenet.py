import os
import torchvision
from torch.utils.data import DataLoader

from .imagenet_classnames import get_classnames


class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data/imagenet'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 **kwargs):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(location, 'val'),
            transform=preprocess,
        )

        self.train_loader = None
        self.val_loader = None
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = get_classnames('openai')
