"""Video dataset module for M3DDM Video Outpainting."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from torch.utils.data import Dataset


@dataclass
class VideoDatasetConfig:
    """Configuration for video dataset.

    Attributes:
        video_dir: Directory containing video files
        max_samples: Maximum number of samples to load (None for all)
        video_extensions: List of video file extensions to search for
        shuffle_on_init: Whether to shuffle files on initialization
    """
    video_dir: str
    max_samples: Optional[int] = None
    video_extensions: tuple = ("*.mp4", "*.avi")
    shuffle_on_init: bool = True


class VideoDataset(Dataset):
    """Dataset for loading video files.

    This dataset recursively searches for video files in a directory
    and returns file paths. The actual video loading is performed
    in the model's forward pass for efficiency.

    Attributes:
        config: VideoDatasetConfig object containing dataset parameters
        files: List of video file paths
    """

    def __init__(self, config: VideoDatasetConfig):
        """Initialize VideoDataset with configuration.

        Args:
            config: VideoDatasetConfig object
        """
        self.config = config
        self.files = self._collect_video_files()

        if config.shuffle_on_init and config.max_samples is not None:
            if len(self.files) > config.max_samples:
                random.shuffle(self.files)
                self.files = self.files[:config.max_samples]
        elif config.max_samples is not None:
            self.files = self.files[:config.max_samples]

    def _collect_video_files(self) -> List[str]:
        """Recursively collect video files from the specified directory.

        Returns:
            List of video file paths

        Raises:
            ValueError: If video_dir does not exist
        """
        if not os.path.exists(self.config.video_dir):
            raise ValueError(f"Video directory does not exist: {self.config.video_dir}")

        files = []
        for root, _, _ in os.walk(self.config.video_dir):
            for ext in self.config.video_extensions:
                for path in Path(root).rglob(ext):
                    files.append(str(path))

        if len(files) == 0:
            raise ValueError(
                f"No video files found in {self.config.video_dir} "
                f"with extensions {self.config.video_extensions}"
            )

        return files

    def __len__(self) -> int:
        """Return the number of video files in the dataset.

        Returns:
            Number of video files
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        """Get video file path by index.

        Args:
            idx: Index of the video file

        Returns:
            Video file path
        """
        return str(self.files[idx])
