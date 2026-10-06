"""
Select a small, patient-level FDG-PET/CT subset from the AutoPET/FDG-PET-CT-Lesions dataset.

Expected input:
    fdg_metadata.csv
The exact column names can vary slightly across dataset versions, so the script
tries to detect common names.

Target:
    35 studies total
      - 30 positive: 10 lung cancer, 10 lymphoma, 10 melanoma
      - 5 negative controls

The script never downloads image data. It only creates a reproducible selection
CSV that can then be used in TCIA/NBIA Data Retriever.

Usage:
    python select_autopet_subset.py --metadata fdg_metadata.csv

Optional:
    python select_autopet_subset.py --metadata fdg_metadata.csv --seed 42
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


POSITIVE_TARGETS = {
    "lung": 10,
    "lymphoma": 10,
    "melanoma": 10,
}
NEGATIVE_TARGET = 5


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {norm_col(c): c for c in df.columns}
    for candidate in candidates:
        key = norm_col(candidate)
        if key in normalized:
            return normalized[key]
    raise KeyError(
        f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
    )


def classify(value: object) -> str:
    s = str(value).strip().lower()
    if any(x in s for x in ["negative", "control", "no lesion", "no lesion"]):
        return "negative"
    if "lymph" in s:
        return "lymphoma"
    if "melanoma" in s:
        return "melanoma"
    if "lung" in s or "nsclc" in s or "non-small" in s:
        return "lung"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Path to fdg_metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="selected_autopet_35.csv")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    df = pd.read_csv(metadata_path)

    patient_col = find_column(
        df,
        ["PatientID", "Patient Id", "Patient_ID", "patient_id"],
    )
    class_col = find_column(
        df,
        ["class", "Class", "study_class", "StudyClass", "diagnosis"],
    )

    # Keep the first study per patient for the compact subset.
    work = df[[patient_col, class_col] + [
        c for c in df.columns
        if norm_col(c) in {
            "age", "sex", "studyinstanceuid", "studyuid", "studyid"
        }
    ]].copy()

    work["label"] = work[class_col].map(classify)
    work = work.drop_duplicates(subset=[patient_col])

    rng = 42

    selected_frames = []
    for label, target in POSITIVE_TARGETS.items():
        subset = work[work["label"] == label].sample(
            n=min(target, len(work[work["label"] == label])),
            random_state=rng,
        )
        selected_frames.append(subset)

    negatives = work[work["label"] == "negative"].sample(
        n=min(NEGATIVE_TARGET, len(work[work["label"] == "negative"])),
        random_state=rng,
    )
    selected_frames.append(negatives)

    selected = pd.concat(selected_frames, ignore_index=True)

    expected = sum(POSITIVE_TARGETS.values()) + NEGATIVE_TARGET
    if len(selected) < expected:
        raise RuntimeError(
            f"Only {len(selected)} studies could be selected; expected {expected}. "
            "Inspect the class labels in the metadata."
        )

    # Deterministic patient-level split.
    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Positive studies: 21 train, 5 validation, 4 test.
    # Negative controls: assigned to the test set so they directly probe false positives.
    positive = selected[selected["label"] != "negative"].copy()
    negative = selected[selected["label"] == "negative"].copy()

    positive["split"] = "train"
    positive.iloc[21:26, positive.columns.get_loc("split")] = "validation"
    positive.iloc[26:, positive.columns.get_loc("split")] = "test"

    negative["split"] = "test"

    final = pd.concat([positive, negative], ignore_index=True)

    # Extra safety: no patient may occur in more than one split.
    split_counts = final.groupby(patient_col)["split"].nunique()
    leakage = split_counts[split_counts > 1]
    if not leakage.empty:
        raise RuntimeError("Patient leakage detected across splits.")

    final = final.sort_values(["split", "label", patient_col]).reset_index(drop=True)
    final.to_csv(args.output, index=False)

    print("\nSelected subset:")
    print(final.groupby(["split", "label"]).size())
    print(f"\nSaved: {args.output}")
    print("\nImportant:")
    print("1. Review the CSV before downloading.")
    print("2. Use the selected PatientIDs/StudyInstanceUIDs to build the TCIA/NBIA manifest.")
    print("3. Keep patient-level splitting; do not randomly split individual slices.")


if __name__ == "__main__":
    main()
