"""CBIS-DDSM metadata and cache utilities refactored from notebooks 02/03."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def uid_from(path: str) -> str:
    """Extract the image UID used by the converted JPEG directory layout."""
    return str(path).replace("\\", "/").split("/")[-2]


def first_jpeg_from_uid(uid: str, jpeg_dir: str | os.PathLike) -> str | None:
    folder = Path(jpeg_dir) / uid
    if not folder.is_dir():
        return None
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return str(file)
    return None


def jpegs_from_metadata_path(metadata_path: str, jpeg_dir: str | os.PathLike) -> list[str]:
    """Resolve a CBIS-DDSM metadata path to all converted image files for its UID."""
    uid = uid_from(metadata_path)
    folder = Path(jpeg_dir) / uid
    if not folder.is_dir():
        return []
    return [
        str(p) for p in sorted(folder.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def join_unique_paths(paths: Iterable[str]) -> str:
    vals = []
    for p in paths:
        s = str(p)
        if s and s != "nan" and s not in vals:
            vals.append(s)
    return "||".join(vals)


def build_img_df(case_csv: str, data_dir: str | os.PathLike, jpeg_dir: str | os.PathLike) -> pd.DataFrame:
    """Build the image-level baseline dataframe used in notebook 02."""
    df = pd.read_csv(Path(data_dir) / case_csv)
    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)
    df["full_jpeg"] = df["image file path"].apply(
        lambda p: first_jpeg_from_uid(uid_from(p), jpeg_dir)
    )
    out = (
        df.dropna(subset=["full_jpeg"])
        .groupby(["patient_id", "left or right breast", "image view"], as_index=False)
        .agg(full_jpeg=("full_jpeg", "first"), label=("label", "max"))
    )
    out["source_csv"] = case_csv
    return out


def row_keys(df: pd.DataFrame) -> np.ndarray:
    return (
        df["patient_id"].astype(str)
        + "|"
        + df["left or right breast"].astype(str)
        + "|"
        + df["image view"].astype(str)
    ).to_numpy()


def preprocess_baseline(df: pd.DataFrame, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Resize mammograms to the 224x224 grayscale baseline representation."""
    X = np.zeros((len(df), size, size), dtype=np.uint8)
    y = df["label"].to_numpy(dtype=np.int64)
    for i, path in enumerate(tqdm(df["full_jpeg"].tolist(), desc="preprocessing")):
        X[i] = np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.uint8)
    return X, y


def load_or_build_metadata(
    case_csv: str,
    cache_name: str,
    data_dir: str | os.PathLike,
    jpeg_dir: str | os.PathLike,
    cache_dir: str | os.PathLike,
) -> pd.DataFrame:
    cache = Path(cache_dir) / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)
    out = build_img_df(case_csv, data_dir, jpeg_dir)
    out.to_csv(cache, index=False)
    return out


def cached_arrays(
    df: pd.DataFrame,
    name: str,
    cache_dir: str | os.PathLike,
    size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached baseline arrays only when row keys and labels still match."""
    cache = Path(cache_dir) / name
    cache.parent.mkdir(parents=True, exist_ok=True)
    expected_keys = row_keys(df)
    expected_y = df["label"].to_numpy(dtype=np.int64)

    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        keys_match = (
            "row_key" in d.files
            and np.array_equal(d["row_key"].astype(str), expected_keys.astype(str))
        )
        labels_match = "y" in d.files and np.array_equal(d["y"], expected_y)
        if keys_match and labels_match:
            return d["X"], d["y"]

    X, y = preprocess_baseline(df, size=size)
    np.savez_compressed(cache, X=X, y=y, row_key=expected_keys)
    return X, y
