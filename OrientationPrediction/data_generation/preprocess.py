"""Preprocessing script: ensures quaternion continuity across sequences.

Follows the methods described in:
https://theorangeduck.com/page/unrolling-rotations
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Column definitions (must match generate_data.py output)
ORIENTATION_COLS = [
    "current_orientation_w",
    "current_orientation_x",
    "current_orientation_y",
    "current_orientation_z",
]

DELTA_COLS = [
    "orientation_change_w",
    "orientation_change_x",
    "orientation_change_y",
    "orientation_change_z",
]


def quat_abs(q):
    """Flip quaternion to positive w hemisphere."""
    return -q if q[0] < 0 else q


def quat_unroll_inplace(rotations):
    """Ensure quaternions stay on same hemisphere throughout sequence."""
    # Make initial rotation be the "short way around"
    rotations[0] = quat_abs(rotations[0])

    # Loop over following rotations
    for i in range(1, len(rotations)):
        # If more than 180 degrees away from previous, flip
        if np.dot(rotations[i], rotations[i - 1]) < 0:
            rotations[i] = -rotations[i]


def preprocess_data(input_path: str, output_path: str, verbose: bool = True) -> int:
    """Ensure quaternion continuity and save preprocessed data. Returns flip count."""
    if verbose:
        print(f"Loading data from: {input_path}")

    df = pd.read_csv(input_path)
    n_samples = len(df)

    if verbose:
        print(f"Loaded {n_samples:,} samples")

    orientations = df[ORIENTATION_COLS].values.astype(np.float64)
    deltas = df[DELTA_COLS].values.astype(np.float64)
    original_orientations = orientations.copy()
    original_deltas = deltas.copy()

    if verbose:
        print("Ensuring quaternion continuity...")

    quat_unroll_inplace(orientations)
    quat_unroll_inplace(deltas)

    # Count how many were flipped
    orientation_flips = np.sum(np.sum(orientations * original_orientations, axis=1) < 0)
    delta_flips = np.sum(np.sum(deltas * original_deltas, axis=1) < 0)

    if verbose:
        print(f"  Orientation quaternions flipped: {orientation_flips:,}")
        print(f"  Delta quaternions flipped: {delta_flips:,}")

    # Update dataframe with processed quaternions
    df[ORIENTATION_COLS] = orientations
    df[DELTA_COLS] = deltas

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed data
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nPreprocessed data saved to: {output_path}")
        print(f"Total samples: {n_samples:,}")
        print(f"Total orientation flips: {orientation_flips:,}")
        print(f"Total delta flips: {delta_flips:,}")

    return int(orientation_flips + delta_flips)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess orientation prediction training data"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV file (raw data from generate_data.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output CSV file (preprocessed data)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
