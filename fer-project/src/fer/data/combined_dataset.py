"""Combined FER dataset adapter merging RAF-DB and FER2013 datasets."""

from pathlib import Path
from typing import Optional

import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split

from fer.data.base_dataset import BaseFERDataset


class CombinedFERDataset(BaseFERDataset):
    """Combined dataset adapter merging preprocessed RAF-DB and FER2013 metadata."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[A.Compose] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Initialize CombinedFERDataset.

        Args:
            data_dir: Root data directory path.
            split: Data split ("train", "val", "test").
            transform: Albumentations transform pipeline.
            val_ratio: Validation split ratio from training data.
            seed: Random seed for stratified splitting.
        """
        self.val_ratio = val_ratio
        self.seed = seed
        super().__init__(data_dir, split, transform)

    def load_metadata(self) -> pd.DataFrame:
        """Merge preprocessed metadata DataFrames from RAF-DB and FER2013.

        Returns:
            pd.DataFrame containing merged image_path, original_label, mapped_label, etc.
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for ds_name in ["rafdb", "fer2013", "affectnet"]:
            csv_p = self.data_dir / "processed" / ds_name / "metadata.csv"
            if csv_p.exists():
                df = pd.read_csv(csv_p)
                tr = df[df["split"] == "train"].reset_index(drop=True)
                vl = df[df["split"] == "val"].reset_index(drop=True)
                ts = df[df["split"] == "test"].reset_index(drop=True)

                if len(tr) > 0:
                    train_dfs.append(tr)
                if len(vl) > 0:
                    val_dfs.append(vl)
                if len(ts) > 0:
                    test_dfs.append(ts)

        if self.split == "test":
            if not test_dfs:
                raise FileNotFoundError("No test metadata found.")
            return pd.concat(test_dfs, ignore_index=True)

        # Merge all training data
        if not train_dfs:
            raise FileNotFoundError("No train metadata found.")
        all_train = pd.concat(train_dfs, ignore_index=True)

        # Create stratified train/val split
        if len(val_dfs) > 0:
            val_combined = pd.concat(val_dfs, ignore_index=True)
        else:
            val_combined = None

        if val_combined is None or len(val_combined) == 0:
            try:
                tr_df, val_df = train_test_split(
                    all_train,
                    test_size=self.val_ratio,
                    random_state=self.seed,
                    stratify=all_train["mapped_label"],
                )
            except ValueError:
                tr_df, val_df = train_test_split(
                    all_train,
                    test_size=self.val_ratio,
                    random_state=self.seed,
                )
        else:
            tr_df = all_train
            val_df = val_combined

        if self.split == "train":
            return tr_df.reset_index(drop=True)
        else:
            return val_df.reset_index(drop=True)
