# datasets/custom_hdf5_regression.py

import os
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

from datasets.utils.continual_dataset import ContinualDataset, store_domain_loaders
from backbone.ResNet18 import resnet18
from utils.conf import base_path_img
from datasets.transforms.denormalization import DeNormalize  # for visualization only


LABEL_COLS = [
    'Vaccum Cleaning',
    'Mopping the Floor',
    'Carry Warm Food',
    'Carry Cold Food',
    'Carry Drinks',
    'Carry Small Objects',
    'Carry Large Objects',
    'Cleaning',
    'Starting a conversation'
]


class HDF5CLDataset(Dataset):
    """
    HDF5-backed dataset for a single (domain, split).

    Assumes HDF5 has groups: {domain}/{split}/images and {domain}/{split}/labels.
    Images: (C, H, W) float32, already resized + ImageNet-normalized.
    Train __getitem__ returns (img, labels_9d, not_aug_img).
    Test __getitem__ returns (img, labels_9d).
    """
    def __init__(self, hdf5_path, domain, split,
                 img_variant="image_path",    # name of the images dataset inside group
                 transform=None,
                 not_aug_transform=None):
        self.hdf5_path = hdf5_path
        self.domain = domain
        self.split = split
        self.img_variant = img_variant
        self.transform = transform
        self.not_aug_transform = not_aug_transform

        self._f = None
        self._length = None

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        if self._length is None:
            with h5py.File(self.hdf5_path, "r") as f:
                grp = f[f"{self.domain}/{self.split}"]
                self._length = len(grp["labels"])
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        dom_split = f"{self.domain}/{self.split}"
        grp = self._f[dom_split]

        imgs_grp = grp["images"]               # group exists as created above
        labels_ds = grp["labels"]

        img_np = imgs_grp[self.img_variant][idx]
        img_tensor = torch.from_numpy(img_np).float()

        # main and not-aug streams identical: no extra transforms
        img = img_tensor
        not_aug_img = img_tensor

        # hooks kept for API symmetry; usually transform is None
        if self.transform is not None:
            img = self.transform(img_tensor)
        if self.not_aug_transform is not None:
            not_aug_img = self.not_aug_transform(img_tensor)

        labels_np = labels_ds[idx]      # shape (9,)
        labels = torch.tensor(labels_np, dtype=torch.float32)

        if self.split == "test":
            # match DN4IL test loader: (img, label)
            return img, labels
        else:
            # match DN4IL train loader: (img, label, not_aug_img)
            return img, labels, not_aug_img


class CustomHDF5Regression(ContinualDataset):
    """
    Domain-incremental regression dataset backed by a single HDF5 file.
    API closely mirrors DN4IL but outputs 9-D regression targets.
    """

    NAME = "custom-hdf5-regression"
    SETTING = "domain-il"
    N_CLASSES_PER_TASK = 1      # dummy, not used for regression
    N_TASKS = 6
    IMG_SIZE = 64
    # just to mirror DN4IL / ImageNet stats; data is already normalized in HDF5
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    DOMAIN_LST = [
        'Home',
        'BigOffice-2',
        'BigOffice-3',
        'Hallway',
        'MeetingRoom',
        'SmallOffice'
    ]

    def __init__(self, args):
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        # No in-graph transforms: HDF5 is already resized + normalized.
        resize_64 = transforms.Resize((self.IMG_SIZE, self.IMG_SIZE))
        self.TRANSFORM = [resize_64]
        self.TRANSFORM_NORM = [resize_64]
        self.TRANSFORM_TEST = [resize_64]
        self.NOT_AUG_TRANSFORM = [resize_64]
        

        data_path = base_path_img()   # analogous to base_path_img()
        self.hdf5_path = os.path.join(data_path, "mean_data_pepper_fold0.hdf5")

    def get_data_loaders(self):
        """
        Mirrors DN4IL.get_data_loaders but uses HDF5CLDataset.
        No extra augmentation; optionally allow aug_norm to plug something later.
        """
        if self.args.aug_norm:
            transform = transforms.Compose(self.TRANSFORM_NORM) if self.TRANSFORM_NORM else None
            test_transform = transforms.Compose(self.TRANSFORM_NORM) if self.TRANSFORM_NORM else None
        else:
            transform = transforms.Compose(self.TRANSFORM) if self.TRANSFORM else None
            test_transform = transforms.Compose(self.TRANSFORM_TEST) if self.TRANSFORM_TEST else None

        not_aug_transform = transforms.Compose(self.NOT_AUG_TRANSFORM) if self.NOT_AUG_TRANSFORM else None

        current_domain = self.DOMAIN_LST[self.i]

        train_dataset = HDF5CLDataset(
            hdf5_path=self.hdf5_path,
            domain=current_domain,
            split="train",
            img_variant="image_path",
            transform=transform,
            not_aug_transform=not_aug_transform
        )

        test_dataset = HDF5CLDataset(
            hdf5_path=self.hdf5_path,
            domain=current_domain,
            split="test",
            img_variant="image_path",
            transform=test_transform,
            not_aug_transform=None
        )

        train_loader, test_loader = store_domain_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    def not_aug_dataloader(self, batch_size):
        # Optional; DN4IL leaves it as pass
        pass

    @staticmethod
    def get_backbone():
        # 9 regression outputs
        return resnet18(9)

    def get_transform(self):
        # identity; data already normalized in HDF5
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_norm_transform():
        # identity; no extra normalization
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_normalization_transform():
        # kept for API completeness; no-op in this setup
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_loss():
        # Regression loss used by DUCA as self.loss
        return nn.MSELoss(reduction="mean")

    @staticmethod
    def get_denormalization_transform():
        # For potential visualization; invert ImageNet normalization
        return DeNormalize(CustomHDF5Regression.MEAN, CustomHDF5Regression.STD)
