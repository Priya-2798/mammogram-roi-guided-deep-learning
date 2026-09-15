"""Training-only categorical metadata encoding used by notebook 07."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_meta_value(value) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    value = str(value).strip()
    return value if value and value.lower() not in {"nan", "none"} else "UNKNOWN"


def norm_value(value) -> str:
    return clean_meta_value(value)


def tokens_from_value(value) -> list[str]:
    value = norm_value(value)
    if value == "UNKNOWN":
        return ["UNKNOWN"]
    return [token for token in value.split("|") if token]


def fit_encoder(df: pd.DataFrame, columns) -> dict[str, list[str]]:
    """Fit category vocabularies on training data only."""
    categories = {}
    for column in columns:
        values = set()
        for value in df[column]:
            values.update(tokens_from_value(value))
        values.add("UNKNOWN")
        categories[column] = sorted(values)
    return categories


def encode(df: pd.DataFrame, categories, columns) -> np.ndarray:
    blocks = []
    for column in columns:
        index = {value: i for i, value in enumerate(categories[column])}
        block = np.zeros((len(df), len(categories[column])), np.float32)
        unknown_i = index["UNKNOWN"]
        for row_i, value in enumerate(df[column]):
            tokens = [
                token if token in index else "UNKNOWN"
                for token in tokens_from_value(value)
            ] or ["UNKNOWN"]
            for token in tokens:
                block[row_i, index.get(token, unknown_i)] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)
