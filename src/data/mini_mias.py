"""mini-MIAS manifest helpers from notebooks 10 and 11."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


MIAS_LINE = re.compile(
    r"^(mdb\d{3})\s+([FGD])\s+"
    r"(CALC|CIRC|SPIC|MISC|ARCH|ASYM|NORM)"
    r"(?:\s+([BM]))?"
)


def build_mias_manifest(
    mias_dir: str | Path,
    info_path: str | Path,
    *,
    require_complete_dataset: bool = True,
) -> pd.DataFrame:
    """Parse mini-MIAS truth data and create patient-grouped normal/abnormal labels.

    The notebook groups consecutive mammograms into 161 patient IDs:
    001/002 -> patient 001, 003/004 -> patient 002, and so on.
    """
    mias_dir = Path(mias_dir)
    info_path = Path(info_path)

    if not mias_dir.is_dir():
        raise FileNotFoundError(f"mini-MIAS image directory not found: {mias_dir}")
    if not info_path.is_file():
        raise FileNotFoundError(f"mini-MIAS truth file not found: {info_path}")

    annotations: dict[str, dict[str, object]] = {}
    with info_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = MIAS_LINE.match(raw_line.strip())
            if not match:
                continue
            ref, tissue, abnormality, severity = match.groups()
            entry = annotations.setdefault(
                ref, {"tissue": tissue, "abnormalities": set(), "severities": set()}
            )
            entry["abnormalities"].add(abnormality)
            if severity:
                entry["severities"].add(severity)

    image_files = sorted(p for p in mias_dir.glob("mdb*.pgm") if p.is_file())
    if require_complete_dataset and len(image_files) != 322:
        raise ValueError(f"Expected 322 mini-MIAS PGM files but found {len(image_files)}")

    rows = []
    for image_path in image_files:
        ref = image_path.stem
        if ref not in annotations:
            raise ValueError(f"No truth-data entry found for {ref}")

        image_number = int(ref.replace("mdb", ""))
        patient_number = (image_number + 1) // 2
        abnormalities = annotations[ref]["abnormalities"]
        if "NORM" in abnormalities and len(abnormalities) > 1:
            raise ValueError(f"{ref} is simultaneously normal and abnormal")

        severities = annotations[ref]["severities"]
        rows.append(
            {
                "image_path": str(image_path),
                "patient_id": f"mias_patient_{patient_number:03d}",
                "label": 0 if abnormalities == {"NORM"} else 1,
                "source": "mini-MIAS",
                "sample_id": ref,
                "side": "right" if image_number % 2 == 1 else "left",
                "tissue": annotations[ref]["tissue"],
                "abnormality": "|".join(sorted(abnormalities)),
                "pathology": "|".join(sorted(severities)) if severities else "normal",
            }
        )

    manifest = pd.DataFrame(rows)
    if require_complete_dataset:
        if len(manifest) != 322:
            raise AssertionError("Manifest should contain exactly 322 image rows.")
        if manifest["patient_id"].nunique() != 161:
            raise AssertionError("Expected exactly 161 patient groups.")
    if not manifest.empty and not manifest["sample_id"].is_unique:
        raise AssertionError("sample_id must be unique.")
    return manifest
