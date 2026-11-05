#!/usr/bin/env python3
"""Convert MRC files, image folders, or stacks to CBF. Inputs stay read-only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import errno
import copy
import glob
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
import tempfile

import fabio
from fabio.cbfimage import CbfImage
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import least_squares

from inputs import MRC_SUFFIXES, MrcSeries, find_inputs, group_inputs, numbered_name
from metadata import geometry

VERSION = "9.1.0"
INT_MAX = np.iinfo(np.int32).max


def bin_frame(frame, factor, bad_pixels=None, saturation=None):
    """Sum nonoverlapping pixels. A bad input makes its whole output bin invalid."""
    height, width = frame.shape
    if height % factor or width % factor:
        raise ValueError(
            "Binning must divide both image dimensions; no pixels are silently discarded"
        )
    invalid = ~np.isfinite(frame)
    if bad_pixels is not None:
        invalid |= bad_pixels
    if saturation is not None:
        invalid |= frame >= saturation
    if (
        frame.dtype.kind in "iu"
        and np.max(np.abs(frame.astype(np.float64))) * factor**2 > 2**53
    ):
        raise ValueError("Input is too large for exact integer conversion")
    values = np.where(invalid, 0, frame).astype(np.float64)
    shape = (height // factor, factor, width // factor, factor)
    values = values.reshape(shape).sum(axis=(1, 3))
    invalid = invalid.reshape(shape).any(axis=(1, 3))
    if not (~invalid).any():
        raise ValueError("Frame contains no valid pixels")
    return values, invalid


def beam_roi(values, invalid, radius):
    height, width = values.shape
    y0, x0 = max(0, height // 2 - radius), max(0, width // 2 - radius)
    y1, x1 = min(height, height // 2 + radius), min(width, width // 2 + radius)
    roi = values[y0:y1, x0:x1].copy()
    valid = ~invalid[y0:y1, x0:x1]
    if np.count_nonzero(valid) < 32:
        raise ValueError(
            "Too few valid pixels in beam search window; specify --beam-center"
        )
    roi[~valid] = np.median(roi[valid])
    return roi, valid, (x0, y0)


def frame_stats(values, invalid, radius):
    peak, signal = 0.0, 0.0
    if radius is not None:
        roi, valid, _ = beam_roi(values, invalid, radius)
        background = float(np.median(values[~invalid][::16]))
        # These measure beam illumination, not diffraction resolution or spot strength.
        peak = float(np.percentile(roi[valid], 99.5) - background)
        signal = float(np.maximum(roi[valid] - background, 0).sum())
    rounded = np.rint(values[~invalid])
    minimum, maximum = float(rounded.min()), float(rounded.max())
    if not np.isfinite([minimum, maximum]).all():
        raise ValueError("Binned intensities exceed the finite storage range")
    return dict(
        peak=peak,
        signal=signal,
        minimum=int(minimum),
        maximum=int(maximum),
        invalid_pixels=int(invalid.sum()),
        rounding_error=float(np.max(np.abs(values[~invalid] - rounded))),
    )


def trim_frames(stats, policy, edge_run):
    """Remove only weak leading/trailing frames; preserve all internal angles."""
    count = len(stats)
    if policy == "none":
        return 0, count
    signals = np.array([[row["peak"], row["signal"]] for row in stats])
    reference = np.percentile(signals, 75, axis=0)
    if not np.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError(
            "Cannot determine illuminated frames; use --trim none and an explicit frame range"
        )
    strong = np.all(signals >= reference * [0.45, 0.65], axis=1)
    length = min(edge_run, count)
    starts = np.flatnonzero(
        np.convolve(strong.astype(int), np.ones(length, dtype=int), "valid") == length
    )
    if not len(starts):
        raise ValueError(
            "No run of strongly illuminated frames; review the movie or use --trim none"
        )
    return int(starts[0]), int(starts[-1] + length)


def find_beam(values, invalid, radius):
    """Fit a broad elliptical halo twice, excluding the direct-beam core."""
    roi, valid, origin = beam_roi(values, invalid, radius)
    smooth = gaussian_filter(median_filter(roi, size=3), 3)
    seed_y, seed_x = np.unravel_index(np.argmax(smooth), smooth.shape)
    background = float(np.percentile(roi[valid], 20))
    noise = max(
        1.0, float(np.median(np.abs(roi[valid] - np.median(roi[valid])))) * 1.4826
    )
    amplitude = float(smooth[seed_y, seed_x] - background)
    if amplitude < 8 * noise:
        return None
    yy, xx = np.indices(roi.shape, dtype=float)
    fit = np.array([background, amplitude, seed_x, seed_y, 8.0, 8.0, 0.0])
    lower = [-np.inf, 0, 1, 1, 1, 1, -0.9]
    upper = [
        np.inf,
        np.inf,
        roi.shape[1] - 2,
        roi.shape[0] - 2,
        radius / 2,
        radius / 2,
        0.9,
    ]
    if not (1 < seed_x < upper[2] and 1 < seed_y < upper[3]):
        return None
    for idx in range(2):
        distance = np.hypot(xx - fit[2], yy - fit[3])
        keep = valid & (distance >= 8) & (distance <= min(radius - 2, 32))
        if keep.sum() < 100:
            return None
        x, y, observed = xx[keep], yy[keep], roi[keep]

        def residual(parameters):
            base, height, cx, cy, sx, sy, rho = parameters
            dx, dy = (x - cx) / sx, (y - cy) / sy
            model = base + height * np.exp(
                -(dx**2 + dy**2 - 2 * rho * dx * dy) / (2 * (1 - rho**2))
            )
            return (model - observed) / noise

        result = least_squares(
            residual, fit, bounds=(lower, upper), loss="soft_l1", max_nfev=100
        )
        if not result.success:
            return None
        fit = result.x
    if np.any(result.active_mask[2:6]) or fit[1] < 8 * noise:
        return None
    return np.array([fit[2] + origin[0], fit[3] + origin[1]])


def robust_center(centers, max_drift):
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 3:
        raise ValueError(
            "Fewer than three reliable beam fits; supply a calibrated --beam-center"
        )
    center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - center, axis=1)
    # Do not hide large beam drift behind smoothing or a median.
    drift = float(np.percentile(distances, 95))
    if drift > max_drift:
        raise ValueError(
            f"Beam varies by {drift:.3f} output pixels; a single XDS origin is inadequate"
        )
    return center, drift


def translate(values, invalid, shift):
    """Integer translation without interpolation, wrapping, or valid padded pixels."""
    dx, dy = map(int, shift)
    height, width = values.shape
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("Requested centering moves the whole detector out of view")
    output = np.zeros_like(values)
    mask = np.ones_like(invalid)
    src = (
        slice(max(0, -dy), min(height, height - dy)),
        slice(max(0, -dx), min(width, width - dx)),
    )
    dst = (
        slice(max(0, dy), min(height, height + dy)),
        slice(max(0, dx), min(width, width + dx)),
    )
    output[dst], mask[dst] = values[src], invalid[src]
    return output, mask


def store_pixels(values, invalid, pedestal):
    rounded = np.rint(values) + pedestal
    valid = rounded[~invalid]
    if not np.isfinite(valid).all() or valid.min() < 0 or valid.max() > INT_MAX:
        raise ValueError(
            "Pedestal/storage range would clip valid pixels; conversion stopped"
        )
    rounded[invalid] = -1
    return rounded.astype(np.int32)


def cbf_header(meta, center, angle, pedestal, overload):
    """Write known values only. Absent geometry must not look like calibration."""
    lines = ["# Detector: MicroED"]
    if meta["pixel_size_mm"] is not None:
        px, py = meta["pixel_size_mm"]
        lines.append(f"# Pixel_size {px / 1000:.12g} m x {py / 1000:.12g} m")
    if meta["distance_mm"] is not None:
        lines.append(f"# Detector_distance {meta['distance_mm'] / 1000:.12g} m")
    if meta["wavelength_a"] is not None:
        lines.append(f"# Wavelength {meta['wavelength_a']:.12g} A")
    if center is not None:
        bx, by = np.asarray(center) + 0.5
        lines.append(f"# Beam_xy ({bx:.8f}, {by:.8f}) pixels")
    for key, value, unit in (
        ("Start_angle", angle, "deg."),
        ("Angle_increment", meta["angle_step_deg"], "deg."),
        ("Exposure_time", meta["exposure_time_s"], "s"),
        ("Exposure_period", meta["frame_time_s"], "s"),
    ):
        if value is not None:
            lines.append(f"# {key} {value:.12g} {unit}")
    lines.extend(
        [f"# Count_cutoff {overload} counts", f"# Storage_pedestal {pedestal}"]
    )
    return "\n".join(lines) + "\n"


def xds_input(meta, center, shape, count, output, pedestal, overload):
    height, width = shape
    px, py = meta["pixel_size_mm"]
    # This is a proper 180-degree rotation about x, not a reflection of y alone.
    axis = np.asarray(meta["rotation_axis"]) * [1, -1, -1]
    bx, by = np.asarray(center) + 1
    return f"""! mrc2cbf {VERSION}; storage pedestal added to valid pixels: {pedestal}
! OFFSET is in stored CBF units. No gain is estimated.
JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
DETECTOR= PILATUS
NX= {width} NY= {height}
QX= {px:.12g} QY= {py:.12g}
OVERLOAD= {overload}
MINIMUM_VALID_PIXEL_VALUE= 0
OFFSET= {pedestal}
! GAIN= 1.0
SENSOR_THICKNESS= 0
AIR= 0
FRACTION_OF_POLARIZATION= -1
TRUSTED_REGION= 0.0 1.42
DIRECTION_OF_DETECTOR_X-AXIS= 1 0 0
DIRECTION_OF_DETECTOR_Y-AXIS= 0 1 0
ORGX= {bx:.8f} ORGY= {by:.8f}
DETECTOR_DISTANCE= {meta["distance_mm"]:.12g}
NAME_TEMPLATE_OF_DATA_FRAMES= {output}/image_??????.cbf
DATA_RANGE= 1 {count}
SPOT_RANGE= 1 {count}
BACKGROUND_RANGE= 1 {count}
ROTATION_AXIS= {axis[0]:.12g} {axis[1]:.12g} {axis[2]:.12g}
STARTING_FRAME= 1
STARTING_ANGLE= {meta["start_angle_deg"]:.12g}
OSCILLATION_RANGE= {meta["angle_step_deg"]:.12g}
X-RAY_WAVELENGTH= {meta["wavelength_a"]:.12g}
INCIDENT_BEAM_DIRECTION= 0 0 1
REFINE(IDXREF)= BEAM AXIS ORIENTATION CELL
REFINE(INTEGRATE)= BEAM ORIENTATION
REFINE(CORRECT)= BEAM ORIENTATION CELL
"""


def dials_helper(output, meta, pedestal, count):
    """Generate a standalone importer that writes only into the caller's directory."""
    return f'''#!/usr/bin/env python3
"""Import this CBF dataset in a writable work directory."""
import argparse
from pathlib import Path
import re
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path({str(output)!r}))
parser.add_argument("--pedestal", type=float, default={pedestal}, help="Exact storage ADU subtracted by DIALS")
gain = parser.add_mutually_exclusive_group()
gain.add_argument("--gain", type=float, default=None, help="Measured ADU per electron")
gain.add_argument("--gain-from-init", type=Path, help="Use the mean gain from a matching XDS INIT.LP, a scalar approximation")
parser.add_argument("--output", default="imported.expt")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("extra", nargs="*", help="Additional dials.import name=value options")
args = parser.parse_args()
if args.gain_from_init:
    text = args.gain_from_init.read_text()
    mean = re.findall(r"MEAN GAIN VALUE\\s*(?:=)?\\s*([0-9.Ee+\\-]+)", text)
    dark = re.findall(r"(?:DARK CURRENT LOOK-UP TABLE IS SET CONSTANT TO|FOR DARK CURRENT IS SET CONSTANT TO\\s+OFFSET=)\\s+([0-9.Ee+\\-]+)", text)
    template = str(args.source.resolve() / "image_??????.cbf")
    if len(mean) != 1 or len(dark) != 1 or float(dark[0]) != args.pedestal or template not in text:
        parser.error("INIT.LP must report one mean gain for this exact CBF path and pedestal")
    args.gain = float(mean[0])
    print("Using XDS mean gain as a scalar approximation; this does not reproduce its spatial gain map.")
if Path(args.output).exists():
    parser.error("Output exists; choose --output with a new name")
if args.gain is not None and not 0 < args.gain < float("inf"):
    parser.error("Gain must be positive and finite")
if not -float("inf") < args.pedestal < float("inf"):
    parser.error("Pedestal must be finite")
files = [args.source / ("image_%06d.cbf" % idx) for idx in range(1, {count + 1})]
if not all(path.is_file() for path in files):
    parser.error("CBF sequence is incomplete; check --source")
command = ["dials.import", *map(str, files), "output.experiments=" + args.output,
           "probe=electron", "geometry.goniometer.axis={",".join(format(v, ".12g") for v in meta["rotation_axis"])}",
           "geometry.detector.panel.pedestal=" + str(args.pedestal)]
if args.gain is not None:
    command.append("geometry.detector.panel.gain=" + str(args.gain))
command.extend(args.extra)
print(shlex.join(command))
if not args.dry_run:
    raise SystemExit(subprocess.run(command).returncode)
'''


def sidecars(args, data):
    """Find a sidecar for this sweep. Do not select an arbitrary XML or MDOC."""
    source = data.paths[0]
    if len(data.paths) == 1:
        matches = {
            "mdoc": [Path(str(source) + ".mdoc"), source.with_suffix(".mdoc")],
            "xml": [source.with_suffix(".xml"), Path(str(source) + ".xml")],
        }
    else:
        prefix = numbered_name(source)[0].rstrip("_.-")
        stems = [source.parent / prefix, source.parent / source.parent.name]
        matches = {
            name: [Path(str(stem) + suffix) for stem in stems for suffix in suffixes]
            for name, suffixes in (
                ("mdoc", (".mdoc", ".mrc.mdoc")),
                ("xml", (".xml", ".mrc.xml")),
            )
        }
    for name in ("mdoc", "xml"):
        selected = getattr(args, name)
        if selected is None:
            candidates = list(
                dict.fromkeys(path for path in matches[name] if path.is_file())
            )
            if len(candidates) == 1:
                selected = candidates[0]
            elif len(candidates) > 1:
                data.notes.append(
                    f"Several {name.upper()} sidecars match; none selected. Use --{name}"
                )
        if selected is not None:
            setattr(args, name, selected.expanduser().resolve(strict=True))


def convert_series(args, data, output):
    started = time.perf_counter()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    sidecars(args, data)
    if args.bad_pixels is not None:
        args.bad_pixels = args.bad_pixels.expanduser().resolve(strict=True)
    sidecar_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.mdoc, args.xml, args.bad_pixels)
        if path is not None
    }
    with data:
        first_number = (
            data.numbers[0][1]
            if data.items[0]["count"] == 1 and data.numbers[0] is not None
            else None
        )
        meta = geometry(
            args,
            data.shape,
            allow_missing=True,
            paths=data.paths,
            first_number=first_number,
        )
        meta["notes"].extend(data.notes)
        factor = args.bin
        if factor is None:
            minimum_factor = max(1, math.ceil(max(data.shape[1:]) / args.target_size))
            factor = next(
                (
                    idx
                    for idx in range(minimum_factor, min(data.shape[1:]) + 1)
                    if data.shape[1] % idx == 0 and data.shape[2] % idx == 0
                ),
                None,
            )
        if factor is None or any(size % factor for size in data.shape[1:]):
            raise ValueError(
                "Cannot sum-bin both axes exactly to this size; choose --bin 1 or another common divisor"
            )
        shape = (data.shape[1] // factor, data.shape[2] // factor)
        meta["input_pixel_size_mm"] = meta["pixel_size_mm"]
        if meta["pixel_size_mm"] is not None:
            meta["pixel_size_mm"] = [value * factor for value in meta["pixel_size_mm"]]
        first, last = args.frames or (1, len(data))
        if not 1 <= first <= last <= len(data):
            raise ValueError(
                "--frames must be an inclusive, 1-based range inside the MRC"
            )
        indices = list(range(first - 1, last))
        bad_pixels = None
        if args.bad_pixels:
            bad_pixels = np.load(args.bad_pixels, allow_pickle=False)
            if bad_pixels.shape != data.shape[1:] or bad_pixels.dtype != np.bool_:
                raise ValueError(
                    "Bad-pixel mask must be a boolean NPY array matching the input frame"
                )
        saturation = args.saturation
        if saturation is None and data.dtype.kind in "iu":
            saturation = float(np.iinfo(data.dtype).max)
            meta["notes"].append(
                "Saturation threshold is the input integer limit; supply detector calibration if lower"
            )
        elif saturation is None:
            meta["notes"].append(
                "No saturation threshold for floating-point input; supply --saturation if known"
            )

        def read_frame(idx):
            frame = data[idx]
            if args.counting:
                valid = np.isfinite(frame)
                if bad_pixels is not None:
                    valid &= ~bad_pixels
                if (frame[valid] < 0).any() or (
                    frame[valid] != np.rint(frame[valid])
                ).any():
                    raise ValueError(
                        "--counting requires nonnegative integer counts in every valid input pixel"
                    )
            return bin_frame(frame, factor, bad_pixels, saturation)

        def inspect_frame(idx):
            values, invalid = read_frame(idx)
            radius = args.beam_radius if args.trim != "none" else None
            try:
                return frame_stats(values, invalid, radius)
            except ValueError:
                if args.trim != "auto":
                    raise
                return frame_stats(values, invalid, None)

        print(
            f"Inspecting {len(indices)} frames; sum bin {factor}; output {shape[1]} x {shape[0]}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            stats = list(pool.map(inspect_frame, indices))
        trim = args.trim
        try:
            left, right = trim_frames(stats, trim, args.edge_run)
        except ValueError:
            if trim != "auto":
                raise
            left, right = 0, len(stats)
            trim = "none"
            meta["notes"].append(
                "Illumination trim was not reliable; all selected frames were kept"
            )
        indices, stats = indices[left:right], stats[left:right]
        sample_indices = sorted(
            set(np.linspace(0, len(indices) - 1, min(21, len(indices)), dtype=int))
        )
        centers = []
        center, drift = None, None
        if args.beam_center is None:
            for idx in sample_indices:
                values, invalid = read_frame(indices[idx])
                try:
                    found = find_beam(values, invalid, args.beam_radius)
                except ValueError:
                    found = None
                if found is not None:
                    centers.append(found)
            if len(centers) >= math.ceil(len(sample_indices) * 0.8):
                if len(sample_indices) < 3 and len(centers) == len(sample_indices):
                    center = np.median(centers, axis=0)
                    drift = float(np.max(np.linalg.norm(centers - center, axis=1)))
                    meta["notes"].append(
                        "Fewer than three frames: beam drift cannot be assessed reliably"
                    )
                    if drift > args.max_beam_drift:
                        center = None
                else:
                    try:
                        center, drift = robust_center(centers, args.max_beam_drift)
                    except ValueError as error:
                        meta["notes"].append(str(error))
            if center is None:
                meta["notes"].append(
                    "Beam centre is unknown; pixels were not moved. Supply --beam-center X Y for processing helpers"
                )
            meta["sources"]["beam_center_xy"] = (
                "halo fits" if center is not None else "missing"
            )
        else:
            # Binned-pixel centres follow the covered input-pixel corners.
            center = (np.asarray(args.beam_center, dtype=float) + 0.5) / factor - 0.5
            meta["sources"]["beam_center_xy"] = "command line"
        if center is not None and (
            not np.isfinite(center).all()
            or not (0 <= center[0] < shape[1] and 0 <= center[1] < shape[0])
        ):
            raise ValueError("Beam centre must lie on this detector")
        if args.center and center is None:
            raise ValueError(
                "Cannot apply --center without a reliable beam centre; use --beam-center X Y"
            )
        shift = (
            np.rint((np.asarray(shape[::-1]) - 1) / 2 - center).astype(int)
            if args.center
            else np.zeros(2, int)
        )
        if center is not None:
            center = center + shift
        minimum = min(row["minimum"] for row in stats)
        required_pedestal = max(0, -minimum)
        pedestal = required_pedestal if args.pedestal is None else args.pedestal
        if pedestal < required_pedestal:
            raise ValueError(
                f"Pedestal {pedestal} would clip pixels; need at least {required_pedestal}"
            )
        if args.counting and pedestal != 0:
            raise ValueError("Counting mode uses zero storage pedestal")
        maximum = max(row["maximum"] for row in stats) + pedestal
        if maximum >= INT_MAX:
            raise ValueError("Stored intensities exceed int32 range")
        overload = maximum + 1  # Saturated input bins were already marked invalid.
        if pedestal:
            meta["notes"].append(
                "XDS OFFSET and DIALS pedestal equal the exact added storage pedestal; this does not repair XDS handling of signed detector noise"
            )
        if meta["start_angle_deg"] is not None:
            if meta["angle_step_deg"] is not None:
                meta["start_angle_deg"] += (
                    data.offsets[indices[0]] * meta["angle_step_deg"]
                )
            elif indices[0] != 0:
                meta["start_angle_deg"] = None
                meta["sources"]["start_angle_deg"] = "missing after frame selection"
        print(
            f"Keeping input frames {indices[0] + 1}-{indices[-1] + 1}; pedestal {pedestal}; beam {center.tolist() if center is not None else 'unknown'}",
            flush=True,
        )
        missing = [
            name
            for name in (
                "pixel_size_mm",
                "distance_mm",
                "wavelength_a",
                "frame_time_s",
                "start_angle_deg",
                "angle_step_deg",
            )
            if meta[name] is None
        ]
        if center is None:
            missing.append("beam_center_xy")
        helper_notes = list(missing)
        if data.offsets != list(range(len(data))):
            helper_notes.append("consecutive input frame numbers")
        if args.cbf_paths:
            helper_notes.append(
                "standard output names; use a directory output for XDS/DIALS helpers"
            )
        if any(char.isspace() or char in "!?*" for char in str(output)):
            helper_notes.append("an output path without spaces or !?* for XDS")
        if args.require_geometry and missing:
            raise ValueError("Processing geometry is incomplete: " + ", ".join(missing))
        output.mkdir(parents=True, exist_ok=False)

        def write_frame(item):
            idx, input_idx = item
            values, invalid = read_frame(input_idx)
            if args.center:
                values, invalid = translate(values, invalid, shift)
            angle = None
            if meta["start_angle_deg"] is not None:
                if meta["angle_step_deg"] is not None:
                    angle = (
                        meta["start_angle_deg"]
                        + (data.offsets[input_idx] - data.offsets[indices[0]])
                        * meta["angle_step_deg"]
                    )
                elif idx == 0:
                    angle = meta["start_angle_deg"]
            pixels = store_pixels(values, invalid, pedestal)
            path = (
                args.cbf_paths[input_idx]
                if args.cbf_paths
                else output / f"image_{idx + 1:06d}.cbf"
            )
            image = CbfImage(data=pixels)
            image.header["_array_data.header_convention"] = "GENERIC_MINI"
            image.header["_array_data.header_contents"] = cbf_header(
                meta, center, angle, pedestal, overload
            )
            # Publish only a verified payload. An existing target cannot be replaced.
            name = (
                path.name
                if path.name.isascii() and not any(char.isspace() for char in path.name)
                else f"image_{idx + 1:06d}.cbf"
            )
            pending = Path(temporary) / name
            image.write(str(pending))
            with fabio.open(str(pending)) as check:
                if not np.array_equal(check.data, pixels):
                    raise IOError(f"CBF round-trip mismatch: {path}")
            try:
                os.link(pending, path)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                    raise
                # Some removable drives do not support hard links.
                with pending.open("rb") as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
            return dict(
                output_frame=idx + 1,
                output_file=str(path),
                input_file=str(data.location(input_idx)[0]),
                input_file_frame=data.location(input_idx)[1] + 1,
                input_frame=input_idx + 1,
                scan_offset=data.offsets[input_idx],
                angle_start_deg=angle,
                invalid_pixels=int(invalid.sum()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )

        with tempfile.TemporaryDirectory(prefix=".write-", dir=output) as temporary:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                written = list(pool.map(write_frame, enumerate(indices)))
        data.check_unchanged()
        if any(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            for path, digest in sidecar_hashes.items()
        ):
            raise ValueError("Metadata or bad-pixel mask changed during conversion")
        if not helper_notes:
            text = xds_input(
                meta, center, shape, len(indices), output, pedestal, overload
            )
            (output / "XDS_20230630.INP").write_text(
                "! Historical full-pedestal reference for XDS 20230630.\n" + text
            )
            (output / "XDS.INP").write_text(text)
            (output / "import_dials.py").write_text(
                dials_helper(output, meta, pedestal, len(indices))
            )
        with (output / "frames.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(written[0]))
            writer.writeheader()
            writer.writerows(written)
        report = dict(
            version=VERSION,
            status="complete",
            source=str(data.paths[0]) if len(data.paths) == 1 else None,
            source_size=data.items[0]["size"] if len(data.paths) == 1 else None,
            source_mtime_ns=data.items[0]["mtime_ns"] if len(data.paths) == 1 else None,
            sources=data.records(),
            metadata_complete=not missing,
            missing_metadata=missing,
            processing_helpers=not helper_notes,
            helper_missing=helper_notes,
            sidecars=sidecar_hashes,
            bad_pixels=str(args.bad_pixels.resolve()) if args.bad_pixels else None,
            input_shape=list(data.shape),
            output_shape=[len(indices), *shape],
            bin=factor,
            input_frames=[indices[0] + 1, indices[-1] + 1],
            trim=trim,
            storage_pedestal=pedestal,
            xds_offset=pedestal,
            saturation=saturation,
            beam_center_xy=center.tolist() if center is not None else None,
            beam_shift_xy=shift.tolist(),
            beam_drift_p95_px=drift,
            beam_fits_xy=[value.tolist() for value in centers],
            max_rounding_error_adu=max(row["rounding_error"] for row in stats),
            geometry=meta,
            command=sys.argv,
            elapsed_seconds=time.perf_counter() - started,
            code_sha256={
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("mrc2cbf.py", "metadata.py", "inputs.py")
            },
        )
        complete = output / "conversion.json.tmp"
        complete.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        complete.replace(output / "conversion.json")
    print(
        f"Verified {len(written)} CBFs in {report['elapsed_seconds']:.1f} s: {output}"
    )
    if report["missing_metadata"]:
        print("Missing metadata: " + ", ".join(report["missing_metadata"]))
    if report["helper_missing"]:
        other = [
            name
            for name in report["helper_missing"]
            if name not in report["missing_metadata"]
        ]
        print(
            "XDS/DIALS helpers not written"
            + (": " + ", ".join(other) if other else " (incomplete geometry)")
        )
    for note in dict.fromkeys(meta["notes"]):
        print(f"Note: {note}")
    return report


def convert(args):
    """Plan all inputs first. Process each sweep with its own geometry and offset."""
    args = copy.copy(args)
    inputs = list(args.inputs)
    output = args.output
    args.cbf_paths = None
    # Keep the published INPUT OUTPUT form, including a quoted input glob.
    if output is None and len(inputs) > 1:
        last = Path(inputs[-1]).expanduser()
        is_input_directory = last.is_dir() and any(
            path.suffix.lower() in MRC_SUFFIXES
            for path in last.iterdir()
            if path.is_file()
        )
        if last.suffix.lower() == ".cbf" or (
            last.suffix.lower() not in MRC_SUFFIXES
            and not glob.has_magic(str(last))
            and not is_input_directory
        ):
            output = last
            inputs.pop()
    paths = find_inputs(inputs)
    groups = group_inputs(paths)
    series = [MrcSeries(items) for items in groups]
    output = (output or Path("cbf")).expanduser().resolve()
    cbf_paths = {}
    if output.suffix.lower() == ".cbf":
        destination = output
        if (
            glob.has_magic(str(destination.parent))
            or any(char in destination.name for char in "?[]")
            or destination.name.count("*") > 1
        ):
            raise ValueError(
                "CBF output accepts one * in the filename, for example 'cbf/*.cbf'"
            )
        if "*" not in destination.name and sum(len(data) for data in series) != 1:
            raise ValueError(
                "One output CBF needs one input frame; use a directory or '*.cbf' for several frames"
            )
        targets = []
        for data in series:
            names = []
            for idx in range(len(data)):
                path, local_idx = data.location(idx)
                item = next(item for item in data.items if item["path"] == path)
                stem = (
                    path.stem
                    if item["count"] == 1
                    else f"{path.stem}_{local_idx + 1:06d}"
                )
                target = destination.with_name(destination.name.replace("*", stem))
                if target.exists() or target.is_symlink():
                    raise ValueError(f"Output already exists: {target}")
                names.append(target)
                targets.append(target)
            cbf_paths[id(data)] = names
        if len({str(path).casefold() for path in targets}) != len(targets):
            raise ValueError("Two input files would write the same CBF output")
        report_name = (
            "mrc2cbf_report"
            if "*" in destination.name
            else destination.name + ".report"
        )
        output = destination.parent / report_name
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if len(series) == 1:
        args.cbf_paths = cbf_paths.get(id(series[0]))
        return convert_series(args, series[0], output)
    if args.mdoc or args.xml:
        raise ValueError(
            "--mdoc and --xml select one sweep; use them with one stack or numbered series"
        )
    names = []
    for data in series:
        stem = (
            data.paths[0].stem
            if len(data.paths) == 1
            else numbered_name(data.paths[0])[0].rstrip("_.-")
        )
        names.append(stem or data.paths[0].parent.name)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(
            "Input sweeps have the same output name; convert them to separate output directories"
        )
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, data in zip(names, series):
        try:
            sweep_args = copy.copy(args)
            sweep_args.cbf_paths = cbf_paths.get(id(data))
            report = convert_series(sweep_args, data, output / name)
            results.append(
                dict(output=name, status="complete", frames=report["output_shape"][0])
            )
        except (ValueError, OSError, RuntimeError) as error:
            print(f"Failed {name}: {error}", file=sys.stderr, flush=True)
            results.append(dict(output=name, status="failed", error=str(error)))
    failed = sum(row["status"] == "failed" for row in results)
    report = dict(
        version=VERSION, status="partial" if failed else "complete", sweeps=results
    )
    (output / "batch.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise ValueError(
            f"{failed} of {len(results)} sweeps failed; see {output / 'batch.json'}"
        )
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="MRC files, directories, or quoted globs; INPUT OUTPUT also works",
    )
    result.add_argument(
        "-o",
        "--output",
        type=Path,
        help="New directory (default: cbf), single .cbf, or quoted '*.cbf' name pattern; never overwritten",
    )
    result.add_argument(
        "--require-geometry",
        action="store_true",
        help="Return an error if processing geometry is incomplete",
    )
    result.add_argument("--version", action="version", version=VERSION)
    result.add_argument("--mdoc", type=Path)
    result.add_argument("--xml", type=Path)
    result.add_argument(
        "--pixel-size-mm",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Override stored MRC pixel sizes; otherwise use MDOC/known detector metadata",
    )
    result.add_argument(
        "--rotation-axis",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the v7 lab-axis fallback in DIALS coordinates (fast +x, slow -y, beam -z)",
    )
    result.add_argument("--distance-mm", type=float)
    wavelength = result.add_mutually_exclusive_group()
    wavelength.add_argument("--wavelength-a", type=float)
    wavelength.add_argument("--voltage-kv", type=float)
    result.add_argument(
        "--start-angle-deg", type=float, help="Start, not centre, of input frame 1"
    )
    result.add_argument("--angle-step-deg", type=float)
    result.add_argument(
        "--frame-time-s", type=float, help="Contiguous exposure period per input frame"
    )
    result.add_argument("--target-size", type=int, default=1024)
    result.add_argument(
        "--bin", type=int, help="Additional spatial SUM binning; overrides target size"
    )
    result.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"))
    result.add_argument(
        "--trim", choices=("auto", "aggressive", "none"), default="auto"
    )
    result.add_argument(
        "--edge-run",
        type=int,
        default=5,
        help="Strong consecutive frames required at retained ends",
    )
    result.add_argument(
        "--beam-center",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Calibrated zero-based input pixel centre",
    )
    result.add_argument(
        "--beam-radius",
        type=int,
        default=64,
        help="Search radius in output pixels about image centre",
    )
    result.add_argument(
        "--max-beam-drift",
        type=float,
        default=2,
        help="Maximum 95th-percentile beam displacement in output pixels",
    )
    result.add_argument(
        "--center",
        action="store_true",
        help="Apply one integer translation to the whole sweep; padded borders invalid",
    )
    result.add_argument(
        "--bad-pixels",
        type=Path,
        help="Boolean input-sized NPY mask; True means invalid",
    )
    result.add_argument(
        "--saturation", type=float, help="Input ADU at/above which a pixel is invalid"
    )
    result.add_argument(
        "--pedestal",
        type=int,
        help="Storage ADU added after binning; default is the minimum that avoids clipping",
    )
    result.add_argument(
        "--counting",
        action="store_true",
        help="Require nonnegative integer counts; force pedestal and XDS offset to zero",
    )
    result.add_argument("--workers", type=int, default=4)
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    for name in ("workers", "target_size", "edge_run", "beam_radius"):
        if getattr(args, name) <= 0:
            cli.error(f"--{name.replace('_', '-')} must be positive")
    if args.beam_radius < 20 or (args.bin is not None and args.bin <= 0):
        cli.error("Beam radius must be at least 20 pixels and binning must be positive")
    if not np.isfinite(args.max_beam_drift) or args.max_beam_drift <= 0:
        cli.error("--max-beam-drift must be positive and finite")
    if args.saturation is not None and (
        not np.isfinite(args.saturation) or args.saturation <= 0
    ):
        cli.error("--saturation must be positive and finite")
    if args.pedestal is not None and args.pedestal < 0:
        cli.error("Pedestal must be nonnegative")
    try:
        convert(args)
    except (ValueError, OSError, RuntimeError) as error:
        cli.exit(
            1,
            f"Error: {error}\nNo complete conversion is certified without conversion.json.\n",
        )


if __name__ == "__main__":
    main()
