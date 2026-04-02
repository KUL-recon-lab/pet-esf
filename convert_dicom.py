#!/usr/bin/env python3
"""
Bulk DICOM to NIfTI converter using dcm2niix.

Converts all DICOM series subfolders found in INPUT_DIR to NIfTI + BIDS sidecar
files, organized under OUTPUT_DIR as:

    <ManufacturerModelName>/<PatientName>_<AcquisitionDate>/<Modality>_<SeriesDescription>_<SeriesNumber>/

Output files are always named image.nii.gz / image.json / image.dcmdump.
A dcmdump sidecar (image.dcmdump) is also written alongside each NIfTI,
containing the full DICOM header dump of the first DICOM file in the series.

Usage:
    python convert_dicom.py /tmp/2026-03__Studies /data/niftis
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pydicom


def find_first_dicom(series_dir: Path) -> Path | None:
    """Return the first readable DICOM file found in a directory."""
    for f in sorted(series_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            pydicom.dcmread(str(f), stop_before_pixels=True)
            return f
        except Exception:
            continue
    return None


def sanitize(value: str) -> str:
    """Strip and replace characters that are unsafe in folder/file names."""
    for ch in r'\/:*?"<>|':
        value = value.replace(ch, "_")
    return value.strip() or "UNKNOWN"


def get_dicom_metadata(dcm_file: Path) -> dict:
    """Extract folder-naming metadata from a DICOM file."""
    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)

    def get(tag: str, default: str = "UNKNOWN") -> str:
        val = getattr(ds, tag, None)
        return str(val).strip() if val else default

    return {
        "manufacturer_model": sanitize(get("ManufacturerModelName")),
        "patient_name":       sanitize(get("PatientName")),
        "acquisition_date":   get("AcquisitionDate", "00000000"),
        "modality":           sanitize(get("Modality")),
        "series_description": sanitize(get("SeriesDescription")),
        "series_number":      get("SeriesNumber", "0"),
    }


def run_dcm2niix(series_dir: Path, output_dir: Path) -> list[Path]:
    """Run dcm2niix on series_dir and return the generated NIfTI paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-b",  "y",      # write BIDS JSON sidecar
        "-ba", "n",      # do NOT anonymize
        "-z",  "y",      # gzip output (.nii.gz)
        "-f",  "image",  # fixed output filename stem
        "-o",  str(output_dir),
        str(series_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] dcm2niix failed:\n{result.stderr.strip()}", file=sys.stderr)
        return []

    niftis = list(output_dir.glob("*.nii.gz")) + list(output_dir.glob("*.nii"))
    return niftis


def run_dcmdump(dcm_file: Path) -> str:
    """Return the dcmdump output for the given DICOM file."""
    result = subprocess.run(
        ["dcmdump", str(dcm_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"# dcmdump failed\n{result.stderr.strip()}\n"
    return result.stdout


def dcmdump_sidecar_path(nifti: Path) -> Path:
    """Derive the image.dcmdump sidecar path for a NIfTI file."""
    # strip both .nii.gz or .nii, then add .dcmdump
    stem = nifti.name.split(".")[0]
    return nifti.parent / f"{stem}.dcmdump"


def convert_series(series_dir: Path, output_root: Path) -> None:
    print(f"Processing: {series_dir.name}")

    first_dcm = find_first_dicom(series_dir)
    if first_dcm is None:
        print(f"  [SKIP] No readable DICOM files found.\n")
        return

    meta = get_dicom_metadata(first_dcm)
    series_folder = f"{meta['modality']}_{meta['series_description']}_{meta['series_number']}"
    dest_dir = (
        output_root
        / meta["manufacturer_model"]
        / f"{meta['patient_name']}_{meta['acquisition_date']}"
        / series_folder
    )

    print(f"  -> {dest_dir.relative_to(output_root)}")

    niftis = run_dcm2niix(series_dir, dest_dir)
    if not niftis:
        print()
        return

    dcmdump_output = run_dcmdump(first_dcm)
    for nifti in sorted(niftis):
        sidecar = dcmdump_sidecar_path(nifti)
        sidecar.write_text(dcmdump_output, encoding="utf-8")
        print(f"  {nifti.name}")
        print(f"  {sidecar.name}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-convert DICOM series folders to NIfTI using dcm2niix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Master folder whose immediate subdirectories are DICOM series.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root output directory for the organized NIfTI files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    series_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not series_dirs:
        print(f"No subdirectories found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(series_dirs)} series folder(s) in {input_dir}\n")
    for series_dir in series_dirs:
        convert_series(series_dir, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
