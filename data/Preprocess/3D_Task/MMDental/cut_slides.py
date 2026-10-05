from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


INPUT_ROOT = Path(r"./MMDental")
OUTPUT_ROOT = Path(r"./cbct_png")

SLICES_PER_VIEW = 20
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5
WINDOW_WIDTH_SCALE = 1.2
GAMMA = 1.0


def choose_indices(axis_length, number):
    if axis_length <= 0:
        raise ValueError("The image contains an empty dimension.")

    start_index = int(round(axis_length * 0.05))
    end_index = int(round(axis_length * 0.95))

    start_index = max(
        0,
        min(start_index, axis_length - 1)
    )

    end_index = max(
        start_index + 1,
        min(end_index, axis_length)
    )

    return np.linspace(
        start_index,
        end_index - 1,
        number,
        dtype=int
    )


def compute_volume_window(volume):
    values = volume[np.isfinite(volume)]
    nonzero_values = values[values != 0]

    if nonzero_values.size > 0:
        values = nonzero_values

    if values.size == 0:
        return 0.0, 1.0

    window_min = float(
        np.percentile(
            values,
            LOW_PERCENTILE
        )
    )

    window_max = float(
        np.percentile(
            values,
            HIGH_PERCENTILE
        )
    )

    if (
        not np.isfinite(window_min)
        or not np.isfinite(window_max)
        or window_max <= window_min
    ):
        window_min = float(values.min())
        window_max = float(values.max())

    if window_max <= window_min:
        window_max = window_min + 1.0

    window_max = (
        window_min
        + (window_max - window_min)
        * WINDOW_WIDTH_SCALE
    )

    return window_min, window_max


def normalize_with_volume_window(
    slice_2d,
    window_min,
    window_max
):
    slice_float = np.asarray(
        slice_2d,
        dtype=np.float32
    )

    finite_mask = np.isfinite(slice_float)

    if not finite_mask.any():
        return np.zeros(
            slice_float.shape,
            dtype=np.uint8
        )

    slice_float = np.nan_to_num(
        slice_float,
        nan=window_min,
        posinf=window_max,
        neginf=window_min
    )

    normalized = (
        (slice_float - window_min)
        / (window_max - window_min)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    normalized = np.power(
        normalized,
        GAMMA
    )

    return (
        normalized * 255.0
    ).astype(np.uint8)


def extract_slice(volume, axis, index):
    if axis == 0:
        return volume[index, :, :]

    if axis == 1:
        return volume[:, index, :]

    if axis == 2:
        return volume[:, :, index]

    raise ValueError(
        f"Unsupported axis: {axis}"
    )


def save_view_slices(
    volume,
    view,
    axis,
    output_dir,
    window_min,
    window_max
):
    indices = choose_indices(
        volume.shape[axis],
        SLICES_PER_VIEW
    )

    for save_index, source_index in enumerate(indices):
        slice_2d = extract_slice(
            volume,
            axis,
            int(source_index)
        )

        slice_2d = np.asarray(
            slice_2d
        ).T

        image_array = normalize_with_volume_window(
            slice_2d,
            window_min,
            window_max
        )

        save_path = (
            output_dir
            / f"{view}_slice_{save_index:03d}.png"
        )

        Image.fromarray(
            image_array
        ).save(save_path)

    print(
        f"    {view}: "
        f"{volume.shape[axis]} original slices, "
        f"indices {indices.tolist()}, "
        f"saved {len(indices)} images"
    )


def find_nifti_file(case_dir):
    preferred_files = [
        case_dir / f"{case_dir.name}.nii.gz",
        case_dir / f"{case_dir.name}.nii"
    ]

    for path in preferred_files:
        if path.is_file():
            return path

    nii_files = sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and (
            path.name.lower().endswith(".nii.gz")
            or path.suffix.lower() == ".nii"
        )
    )

    if not nii_files:
        print(
            f"[SKIPPED] {case_dir.name}: "
            f"No NIfTI file found"
        )
        return None

    if len(nii_files) > 1:
        print(
            f"[SKIPPED] {case_dir.name}: "
            f"Multiple NIfTI files found"
        )
        return None

    return nii_files[0]


def process_case(case_dir):
    nifti_path = find_nifti_file(
        case_dir
    )

    if nifti_path is None:
        return False

    print(
        f"[PROCESSING] {case_dir.name} -> "
        f"{nifti_path.name}"
    )

    nifti_image = nib.load(
        str(nifti_path)
    )

    canonical_image = nib.as_closest_canonical(
        nifti_image
    )

    volume = canonical_image.get_fdata(
        dtype=np.float32
    )

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if volume.ndim == 4 and volume.shape[3] == 1:
        volume = volume[:, :, :, 0]
    elif volume.ndim != 3:
        raise ValueError(
            f"Only 3D NIfTI images are supported. "
            f"Current data shape: {volume.shape}"
        )

    window_min, window_max = compute_volume_window(
        volume
    )

    print(
        f"    volume window: "
        f"{window_min:.6f} to "
        f"{window_max:.6f}, "
        f"window width scale: "
        f"{WINDOW_WIDTH_SCALE}, "
        f"gamma: {GAMMA}"
    )

    case_output_dir = (
        OUTPUT_ROOT
        / case_dir.name
    )

    case_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_view_slices(
        volume,
        "axial",
        2,
        case_output_dir,
        window_min,
        window_max
    )

    save_view_slices(
        volume,
        "coronal",
        1,
        case_output_dir,
        window_min,
        window_max
    )

    save_view_slices(
        volume,
        "sagittal",
        0,
        case_output_dir,
        window_min,
        window_max
    )

    print(
        f"[COMPLETED] {case_dir.name}: "
        f"Saved 60 images to "
        f"{case_output_dir}"
    )

    return True


def main():
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Input root directory does not exist: "
            f"{INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_resolved = OUTPUT_ROOT.resolve()

    case_dirs = sorted(
        path
        for path in INPUT_ROOT.iterdir()
        if path.is_dir()
        and path.resolve() != output_resolved
    )

    if not case_dirs:
        print(
            f"No subdirectories found in: "
            f"{INPUT_ROOT}"
        )
        return

    completed = 0
    failed = 0

    for case_dir in case_dirs:
        try:
            if process_case(case_dir):
                completed += 1
        except Exception as error:
            failed += 1

            print(
                f"[FAILED] {case_dir.name}: "
                f"{error}"
            )

    print(
        f"\nProcessing finished: "
        f"{completed} completed, "
        f"{failed} failed"
    )


if __name__ == "__main__":
    main()
