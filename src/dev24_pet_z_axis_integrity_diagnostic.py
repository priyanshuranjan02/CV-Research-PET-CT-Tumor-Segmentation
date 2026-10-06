"""
Dev24 — PET Z-Axis Integrity Diagnostic

Purpose:
    Investigate the abrupt PET disappearance in
    PETCT_185da4c8b6 between z=299 and z=300.

    Questions:
        1. Does the original PET contain the same collapse?
        2. Does the collapse appear only after Dev10 loading/alignment?
        3. What is the PET intensity profile around the transition?

    Dev10 is NOT modified.
"""

from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

RESULT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev24_pet_z_axis_integrity_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


CASE_NAME = "PETCT_185da4c8b6"

Z_START = 270
Z_END = 320


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_stats(volume, z):
    """
    Calculate robust statistics for one z slice.
    """

    if z < 0 or z >= volume.shape[0]:
        return {
            "z": z,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "positive_voxels": 0,
            "ge_2p25": 0,
            "ge_3": 0,
        }

    sl = volume[z].astype(np.float32)

    return {
        "z": int(z),
        "min": float(np.min(sl)),
        "max": float(np.max(sl)),
        "mean": float(np.mean(sl)),
        "median": float(np.median(sl)),
        "positive_voxels": int(np.sum(sl > 0)),
        "ge_2p25": int(np.sum(sl >= 2.25)),
        "ge_3": int(np.sum(sl >= 3.0)),
    }


def find_case_files():
    """
    Search the project data area for files associated with the case.
    """

    data_roots = [
        ROOT / "data",
        ROOT / "dataset",
        ROOT / "datasets",
    ]

    extensions = {
        ".nii",
        ".gz",
        ".mha",
        ".mhd",
        ".nrrd",
        ".npy",
        ".npz",
    }

    matches = []

    for root in data_roots:

        if not root.exists():
            continue

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            path_string = str(path).lower()

            if CASE_NAME.lower() not in path_string:
                continue

            if (
                path.suffix.lower() in extensions
                or path.name.lower().endswith(".nii.gz")
            ):
                matches.append(path)

    return sorted(
        set(matches)
    )


def identify_pet_candidates(paths):
    """
    Identify likely PET/SUV files by filename.
    """

    tokens = [
        "pet",
        "suv",
        "fdg",
    ]

    candidates = []

    for path in paths:

        name = path.name.lower()

        if any(
            token in name
            for token in tokens
        ):
            candidates.append(path)

    return candidates


def load_sitk_volume(path):
    """
    Load a SimpleITK-supported volume.
    """

    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    return image, array


def analyze_loaded_pet():
    """
    Analyze PET after Dev10's own loading/alignment.
    """

    print_header(
        "DEV10 LOADED / ALIGNED PET"
    )

    case = dev10.load_case(
        CASE_NAME
    )

    suv = case["suv"]

    print(
        f"Loaded PET shape: {suv.shape}"
    )

    print(
        f"Global PET max: "
        f"{np.max(suv):.6f}"
    )

    rows = []

    start = max(
        0,
        Z_START
    )

    end = min(
        suv.shape[0] - 1,
        Z_END
    )

    for z in range(
        start,
        end + 1
    ):

        rows.append(
            safe_stats(
                suv,
                z
            )
        )

    df = pd.DataFrame(
        rows
    )

    print(
        df.to_string(
            index=False
        )
    )

    return df


def analyze_original_candidates(
    candidates
):

    print_header(
        "ORIGINAL PET CANDIDATES"
    )

    if not candidates:

        print(
            "No PET/SUV candidate files were "
            "automatically identified."
        )

        print(
            "\nThis means the dataset layout "
            "needs manual inspection."
        )

        return []

    results = []

    for path in candidates:

        print(
            f"\nCandidate: {path}"
        )

        try:

            image, volume = (
                load_sitk_volume(
                    path
                )
            )

            print(
                f"Shape: {volume.shape}"
            )

            print(
                f"Spacing: "
                f"{image.GetSpacing()}"
            )

            print(
                f"Origin: "
                f"{image.GetOrigin()}"
            )

            print(
                f"Direction: "
                f"{image.GetDirection()}"
            )

            print(
                f"Global min: "
                f"{np.min(volume):.6f}"
            )

            print(
                f"Global max: "
                f"{np.max(volume):.6f}"
            )

            # SimpleITK GetArrayFromImage is z,y,x.
            rows = []

            start = max(
                0,
                Z_START
            )

            end = min(
                volume.shape[0] - 1,
                Z_END
            )

            for z in range(
                start,
                end + 1
            ):

                row = safe_stats(
                    volume,
                    z
                )

                row["file"] = str(path)

                rows.append(
                    row
                )

            results.append(
                pd.DataFrame(
                    rows
                )
            )

        except Exception as exc:

            print(
                f"Could not read: {exc}"
            )

    return results


def create_loaded_pet_plot(
    df,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        df["z"],
        df["max"],
        marker="o",
        linewidth=1.8,
        label="PET max"
    )

    ax.plot(
        df["z"],
        df["mean"],
        marker="s",
        linewidth=1.5,
        label="PET mean"
    )

    ax.plot(
        df["z"],
        df["ge_2p25"],
        marker="^",
        linewidth=1.5,
        label="Voxels >= 2.25"
    )

    ax.axvline(
        299,
        linestyle="--",
        linewidth=1.2,
        label="z=299"
    )

    ax.axvline(
        300,
        linestyle="--",
        linewidth=1.2,
        label="z=300"
    )

    ax.set_xlabel(
        "Loaded PET z"
    )

    ax.set_ylabel(
        "Value / voxel count"
    )

    ax.set_title(
        "Dev24 — Dev10 Loaded PET Z-Axis Profile"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    print_header(
        "DEV24 — PET Z-AXIS INTEGRITY DIAGNOSTIC"
    )

    print(
        f"Case: {CASE_NAME}"
    )

    # ------------------------------------------------
    # 1. Analyze Dev10 loaded PET
    # ------------------------------------------------

    loaded_df = analyze_loaded_pet()

    loaded_csv = (
        RESULT_DIR /
        "dev24_loaded_pet_z_profile.csv"
    )

    loaded_df.to_csv(
        loaded_csv,
        index=False
    )

    loaded_plot = (
        RESULT_DIR /
        "dev24_loaded_pet_z_profile.png"
    )

    create_loaded_pet_plot(
        loaded_df,
        loaded_plot
    )

    # ------------------------------------------------
    # 2. Search for original PET
    # ------------------------------------------------

    print_header(
        "SEARCHING DATASET FOR ORIGINAL PET"
    )

    all_case_files = find_case_files()

    print(
        f"Case-related image files found: "
        f"{len(all_case_files)}"
    )

    for path in all_case_files:

        print(
            f"  {path}"
        )

    pet_candidates = (
        identify_pet_candidates(
            all_case_files
        )
    )

    print(
        f"\nLikely PET/SUV files: "
        f"{len(pet_candidates)}"
    )

    for path in pet_candidates:

        print(
            f"  {path}"
        )

    # ------------------------------------------------
    # 3. Analyze original PET candidates
    # ------------------------------------------------

    original_results = (
        analyze_original_candidates(
            pet_candidates
        )
    )

    if original_results:

        original_df = pd.concat(
            original_results,
            ignore_index=True
        )

        original_csv = (
            RESULT_DIR /
            "dev24_original_pet_z_profiles.csv"
        )

        original_df.to_csv(
            original_csv,
            index=False
        )

        print(
            f"\nOriginal PET profile saved:"
            f"\n{original_csv}"
        )

    else:

        original_csv = None

    # ------------------------------------------------
    # 4. Explicit transition report
    # ------------------------------------------------

    print_header(
        "Z=299 → Z=300 TRANSITION"
    )

    for z in [
        297,
        298,
        299,
        300,
        301,
        302,
    ]:

        row = loaded_df[
            loaded_df["z"] == z
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        print(
            f"z={z}: "
            f"max={row['max']:.6f}, "
            f"mean={row['mean']:.6f}, "
            f">=2.25={int(row['ge_2p25'])}, "
            f">=3={int(row['ge_3'])}"
        )

    # ------------------------------------------------
    # 5. Final output
    # ------------------------------------------------

    print_header(
        "DEV24 COMPLETE"
    )

    print(
        f"Loaded PET CSV:"
        f"\n{loaded_csv}"
    )

    print(
        f"\nLoaded PET plot:"
        f"\n{loaded_plot}"
    )

    if original_csv:

        print(
            f"\nOriginal PET CSV:"
            f"\n{original_csv}"
        )

    print(
        "\nIMPORTANT:"
        "\nDev10 was not modified."
    )


if __name__ == "__main__":
    main()