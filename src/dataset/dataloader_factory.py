"""DataLoader factory for M3DDM Video Outpainting."""

import os
from dataclasses import dataclass
from typing import Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .video_dataset import VideoDataset, VideoDatasetConfig


@dataclass
class DataloaderConfig:
    """Configuration for dataloaders.

    Attributes:
        data_dir: Root directory containing 'train' and 'val' subdirectories
        batch_size: Batch size for training and validation
        num_workers: Number of worker processes for data loading
        max_samples: Maximum number of samples to load (None for all)
        train_shuffle: Whether to shuffle training data
        val_shuffle: Whether to shuffle validation data
    """
    data_dir: str
    batch_size: int = 1
    num_workers: int = 0
    max_samples: int = None
    train_shuffle: bool = True
    val_shuffle: bool = False


class VideoDataModule(pl.LightningDataModule):
    """Lightning DataModule for video datasets.

    This module manages train and validation datasets and their dataloaders.
    It expects the data_dir to contain 'train' and 'val' subdirectories.

    Attributes:
        config: DataloaderConfig object containing dataloader parameters
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """

    def __init__(self, config: DataloaderConfig):
        """Initialize VideoDataModule with configuration.

        Args:
            config: DataloaderConfig object
        """
        super().__init__()
        self.config = config
        self.train_dir = os.path.join(config.data_dir, 'train')
        self.val_dir = os.path.join(config.data_dir, 'val')

        # Validate directories
        if not os.path.exists(self.train_dir):
            raise ValueError(f"Training directory does not exist: {self.train_dir}")
        if not os.path.exists(self.val_dir):
            raise ValueError(f"Validation directory does not exist: {self.val_dir}")

    def setup(self, stage: str = None):
        """Setup datasets for training and validation.

        Args:
            stage: Stage of training ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            # Setup training dataset
            train_config = VideoDatasetConfig(
                video_dir=self.train_dir,
                max_samples=self.config.max_samples,
                shuffle_on_init=True
            )
            self.train_dataset = VideoDataset(train_config)

            # Setup validation dataset
            val_config = VideoDatasetConfig(
                video_dir=self.val_dir,
                max_samples=self.config.max_samples,
                shuffle_on_init=False
            )
            self.val_dataset = VideoDataset(val_config)

            print(f"Training dataset: {len(self.train_dataset)} videos")
            print(f"Validation dataset: {len(self.val_dataset)} videos")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            Training DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.train_shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            Validation DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.val_shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0
        )
